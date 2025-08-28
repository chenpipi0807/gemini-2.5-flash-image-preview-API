import os
import base64
import mimetypes
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError
import base64
from PIL import Image
import io
import socket
import threading
import subprocess
import re
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'tiff', 'ico', 'heic', 'heif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_api_key():
    """Get API key from environment variable or key.txt file"""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        try:
            with open('key.txt', 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            return None
    return api_key

def save_binary_file(file_name, data):
    """Save binary data to file"""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")

def calculate_tokens_and_cost(prompt, uploaded_files, generated_images, model_name):
    """Calculate input/output tokens and cost based on Gemini API pricing"""
    
    # Token counting
    input_tokens = 0
    output_tokens = 0
    
    # Text input tokens (approximately 4 characters = 1 token)
    if prompt:
        input_tokens += len(prompt) // 4
    
    # Image input tokens (1024x1024 = 1290 tokens, scale for other sizes)
    for file in uploaded_files:
        if file and file.filename != '':
            # Estimate based on file size - rough approximation
            # For now, assume average uploaded image = 1290 tokens
            input_tokens += 1290
    
    # Image output tokens (1024x1024 = 1290 tokens each)
    output_tokens = len(generated_images) * 1290
    
    # Cost calculation based on Gemini 2.5 Flash Image Preview pricing
    # These are approximate rates - adjust based on actual model pricing
    cost_per_1k_input = 0.0  # Input is often free or very cheap for image models
    cost_per_1k_output = 0.03  # $0.03 per 1000 tokens for image output
    
    input_cost = (input_tokens / 1000) * cost_per_1k_input
    output_cost = (output_tokens / 1000) * cost_per_1k_output
    total_cost = input_cost + output_cost
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'input_cost_usd': round(input_cost, 6),
        'output_cost_usd': round(output_cost, 6),
        'total_cost_usd': round(total_cost, 6),
        'cost_breakdown': {
            'input_text_tokens': len(prompt) // 4 if prompt else 0,
            'input_image_tokens': input_tokens - (len(prompt) // 4 if prompt else 0),
            'output_image_tokens': output_tokens,
            'rate_per_1k_input': cost_per_1k_input,
            'rate_per_1k_output': cost_per_1k_output
        }
    }

def get_error_message(error_code, error_message):
    """Get user-friendly error message based on Gemini API error codes"""
    error_messages = {
        400: "请求参数无效。请检查您的输入参数是否正确。",
        401: "API密钥无效或已过期。请检查您的API密钥设置。",
        403: "权限被拒绝。您的API密钥可能没有访问此功能的权限。",
        404: "请求的模型或资源不存在。请检查模型名称是否正确。",
        429: "请求频率过高或配额已用完。请稍后重试或检查您的配额限制。",
        500: "服务器内部错误。这是Google服务端的问题，请稍后重试。",
        503: "服务暂时不可用。Google服务可能正在维护，请稍后重试。",
        504: "请求超时。您的请求处理时间过长，请尝试简化请求或稍后重试。"
    }
    
    # 特殊错误类型检查
    if "RESOURCE_EXHAUSTED" in error_message:
        return "配额已用完。您的免费配额或付费配额已达到限制，请等待重置或升级计划。"
    elif "PERMISSION_DENIED" in error_message:
        return "权限被拒绝。请检查您的API密钥权限或项目设置。"
    elif "INVALID_ARGUMENT" in error_message:
        return "请求参数无效。请检查您的提示词、图片格式或其他参数设置。"
    elif "FAILED_PRECONDITION" in error_message:
        return "请求条件不满足。请检查模型要求和输入格式。"
    elif "UNAUTHENTICATED" in error_message:
        return "身份验证失败。请检查您的API密钥是否正确设置。"
    elif "CANCELLED" in error_message:
        return "请求被取消。可能是网络问题或请求被中断。"
    elif "DEADLINE_EXCEEDED" in error_message:
        return "请求超时。请尝试减少输入内容或稍后重试。"
    elif "UNAVAILABLE" in error_message:
        return "服务暂时不可用。请稍后重试。"
    elif "safety" in error_message.lower() or "blocked" in error_message.lower():
        return "内容被安全过滤器阻止。请修改您的提示词，避免可能违反内容政策的内容。"
    elif "recitation" in error_message.lower():
        return "内容可能涉及版权问题。请使用更原创的提示词。"
    
    # 根据错误代码返回通用消息
    return error_messages.get(error_code, f"未知错误 (代码: {error_code}): {error_message}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_content():
    """Main API endpoint for generating content with Gemini"""
    start_time = time.time()  # 开始计时
    
    try:
        api_key = get_api_key()
        if not api_key:
            return jsonify({'error': 'API key not found. Please set GEMINI_API_KEY environment variable or create key.txt file.'}), 400

        # Get parameters from form with safe conversion
        prompt = request.form.get('prompt', '')
        model_name = request.form.get('model', 'gemini-2.5-flash-image-preview')
        
        try:
            temperature = float(request.form.get('temperature', '1.0') or '1.0')
        except (ValueError, TypeError):
            temperature = 1.0
            
        try:
            max_output_tokens = int(request.form.get('max_output_tokens', '8192') or '8192')
        except (ValueError, TypeError):
            max_output_tokens = 8192
            
        try:
            top_p = float(request.form.get('top_p', '0.95') or '0.95')
        except (ValueError, TypeError):
            top_p = 0.95
            
        try:
            top_k = int(request.form.get('top_k', '40') or '40')
        except (ValueError, TypeError):
            top_k = 40
            
        try:
            seed = int(request.form.get('seed', '42') or '42')
        except (ValueError, TypeError):
            seed = 42
        
        # Safety settings - use form values with reasonable defaults
        safety_settings_list = [
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold=request.form.get('safety_harm_category_harassment', 'BLOCK_NONE')
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold=request.form.get('safety_harm_category_hate_speech', 'BLOCK_NONE')
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold=request.form.get('safety_harm_category_sexually_explicit', 'BLOCK_NONE')
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold=request.form.get('safety_harm_category_dangerous_content', 'BLOCK_NONE')
            )
        ]

        # Initialize client
        client = genai.Client(api_key=api_key)

        # Handle uploaded images first to check count
        uploaded_files = request.files.getlist('images')
        
        # Build content parts
        parts = []
        
        # Add text part if provided
        if prompt:
            # Different prompts for text-to-image vs image editing
            if len(uploaded_files) == 0:
                # Pure text-to-image: use stronger generation commands
                enhanced_prompt = f"Generate an image: {prompt}\n\nIMPORTANT: You must output an actual image file, not just text description. Create and return the visual content as an image."
            else:
                # Image editing: use modification commands
                enhanced_prompt = f"{prompt}\n\n请直接生成修改后的图像，不要只提供文字说明。"
            parts.append(types.Part.from_text(text=enhanced_prompt))
        print(f"📁 处理上传文件: {len(uploaded_files)} 个文件")
        
        for i, file in enumerate(uploaded_files):
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_size = len(file.read())
                file.seek(0)  # Reset file pointer
                
                print(f"  - 文件 {i+1}: {file.filename} ({file_size/1024/1024:.2f} MB)")
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Read and encode image
                with open(file_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                # Get MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = 'image/jpeg'  # Default fallback
                
                # Add image part
                parts.append(types.Part.from_bytes(
                    mime_type=mime_type,
                    data=image_data
                ))
                
                # Clean up uploaded file
                os.remove(file_path)

        if not parts:
            return jsonify({'error': 'No content provided. Please provide text prompt or upload images.'}), 400

        # Create content
        content = types.Content(role="user", parts=parts)

        # 图像生成模式 - 让模型自然决定输出格式
        print("🎨 图像生成模式")
        
        # Debug: Print API call parameters
        print("🚀 API调用参数:")
        print(f"  - 模型: {model_name}")
        print(f"  - 响应模式: 自动")
        print(f"  - 提示词: {prompt[:50]}...")
        print(f"  - 上传文件数: {len([f for f in uploaded_files if f and f.filename != ''])}")
        print(f"  - 随机种子: {seed}")
        print(f"  - 温度: {temperature}")
        print(f"  - 最大输出: {max_output_tokens}")
        
        start_time = time.time()
        
        # For pure text-to-image, force image output
        config_params = {
            'temperature': temperature,
            'seed': seed,
            'max_output_tokens': max_output_tokens,
            'safety_settings': safety_settings_list
        }
        
        # Force image generation for text-to-image requests
        if len([f for f in uploaded_files if f and f.filename != '']) == 0:
            config_params['response_modalities'] = ['IMAGE']
            print("  - 强制图像输出模式: 已启用")
        
        response = client.models.generate_content(
            model=model_name,
            contents=[content],  
            config=types.GenerateContentConfig(**config_params)
        )
        api_call_duration = time.time() - start_time  

        # 处理非流式响应 - 只处理图像输出
        response_text = ""  # 不需要文本响应
        generated_images = []
        file_index = 0
        text_responses = []  # 收集文本响应用于错误分析
        
        print(f"🔍 开始处理API响应...")
        
        # 检查响应结构
        if (response.candidates and 
            response.candidates[0].content and 
            response.candidates[0].content.parts):
            
            for i, part in enumerate(response.candidates[0].content.parts):
                print(f"  - Part {i}: {type(part)}")
                
                # 只处理图像部分
                if (hasattr(part, 'inline_data') and part.inline_data and 
                    hasattr(part.inline_data, 'data') and part.inline_data.data):
                    try:
                        print(f"🎨 发现图像数据！")
                        
                        # 保存图像
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        filename = f'generated_image_{timestamp}_{file_index}.png'
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        inline_data = part.inline_data
                        data_buffer = inline_data.data
                        
                        # 根据MIME类型确定文件扩展名
                        if hasattr(inline_data, 'mime_type') and inline_data.mime_type:
                            file_extension = mimetypes.guess_extension(inline_data.mime_type)
                            if file_extension:
                                filename = f'generated_image_{timestamp}_{file_index}{file_extension}'
                                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        # 直接保存二进制数据
                        with open(filepath, 'wb') as f:
                            f.write(data_buffer)
                        
                        file_size = os.path.getsize(filepath)
                        print(f"✅ 图片已保存: {filepath} ({file_size} 字节)")
                        
                        if file_size > 100:  # 有效文件
                            generated_images.append(filename)
                            file_index += 1
                        else:
                            print(f"❌ 文件太小，删除: {filepath}")
                            os.remove(filepath)
                            
                    except Exception as img_error:
                        print(f"❌ 图片保存失败: {img_error}")
                
                # 收集文本响应用于错误分析
                elif hasattr(part, 'text') and part.text:
                    text_responses.append(part.text)
                    print(f"📝 收到文本响应: {part.text[:50]}...")
        
        # 调试信息
        print(f"🔍 调试信息:")
        print(f"  - 生成图片数量: {len(generated_images)}")
        print(f"  - 图片文件名: {generated_images}")
        print(f"  - 文本响应数量: {len(text_responses)}")
        
        # 如果没有生成图像但有文本响应，可能是正常的文本对话
        if not generated_images:
            print(f"❌ 没有生成任何图像")
            
            # 如果有文本响应，显示文本内容
            if text_responses:
                response_text = " ".join(text_responses)
                print(f"📝 收到文本响应: {response_text}")
                
                # 返回文本响应而不是错误
                total_duration = time.time() - start_time
                
                # Calculate tokens and cost for text response
                token_cost_info = calculate_tokens_and_cost(prompt, uploaded_files, [], model_name)
                # Add text output tokens (approximate)
                text_output_tokens = len(response_text) // 4
                token_cost_info['output_tokens'] = text_output_tokens
                token_cost_info['total_tokens'] = token_cost_info['input_tokens'] + text_output_tokens
                token_cost_info['output_cost_usd'] = 0.0  # Text output is typically free or very cheap
                token_cost_info['total_cost_usd'] = token_cost_info['input_cost_usd']
                
                result = {
                    'success': True,
                    'response': response_text,
                    'generated_images': [],
                    'generated_image_urls': [],
                    'model': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'timing': {
                        'total_duration': round(total_duration, 2),
                        'api_call_duration': round(api_call_duration, 2),
                        'processing_duration': round(total_duration - api_call_duration, 2)
                    },
                    'tokens': token_cost_info,
                    'config': {
                        'temperature': temperature,
                        'max_output_tokens': max_output_tokens,
                        'top_p': top_p,
                        'top_k': top_k,
                        'seed': seed
                    }
                }
                return jsonify(result)
            else:
                # 真正的错误情况：既没有图像也没有文本
                total_duration = time.time() - start_time
                return jsonify({
                    'success': False,
                    'error': '模型没有返回任何内容，请重试',
                    'error_code': 500,
                    'timing': {
                        'total_duration': round(total_duration, 2),
                        'api_call_duration': round(api_call_duration, 2),
                        'processing_duration': round(total_duration - api_call_duration, 2)
                    }
                }), 500
        
        # Convert local filenames to full URLs
        base_url = request.url_root.rstrip('/')
        generated_image_urls = [f"{base_url}/uploads/{filename}" for filename in generated_images]
        
        total_duration = time.time() - start_time  # 总耗时
        
        # Calculate tokens and cost
        token_cost_info = calculate_tokens_and_cost(prompt, uploaded_files, generated_images, model_name)
        
        result = {
            'success': True,
            'response': '',  # 不返回文本响应
            'generated_images': generated_images,
            'generated_image_urls': generated_image_urls,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'timing': {
                'total_duration': round(total_duration, 2),
                'api_call_duration': round(api_call_duration, 2),
                'processing_duration': round(total_duration - api_call_duration, 2)
            },
            'tokens': token_cost_info,
            'config': {
                'temperature': temperature,
                'max_output_tokens': max_output_tokens,
                'top_p': top_p,
                'top_k': top_k,
                'seed': seed
            }
        }
        
        print(f"📤 返回结果: {len(generated_images)} 张图片")
        print(f"⏱️ 总耗时: {total_duration:.2f}秒 (API调用: {api_call_duration:.2f}秒)")

        return jsonify(result)

    except APIError as e:
        total_duration = time.time() - start_time
        error_code = getattr(e, 'code', 500)
        error_message = str(e)
        user_friendly_message = get_error_message(error_code, error_message)
        
        print(f"❌ Gemini API错误 (代码: {error_code}): {error_message}")
        print(f"⏱️ 失败耗时: {total_duration:.2f}秒")
        
        return jsonify({
            'error': user_friendly_message,
            'error_code': error_code,
            'error_type': 'api_error',
            'original_error': error_message,
            'timing': {
                'total_duration': round(total_duration, 2)
            }
        }), error_code if error_code in [400, 401, 403, 404, 429] else 500
        
    except Exception as e:
        total_duration = time.time() - start_time
        print(f"❌ 系统错误: {e}")
        print(f"⏱️ 失败耗时: {total_duration:.2f}秒")
        
        return jsonify({
            'error': f'系统错误: {str(e)}',
            'error_code': 500,
            'error_type': 'system_error',
            'timing': {
                'total_duration': round(total_duration, 2)
            }
        }), 500

@app.route('/api/models')
def get_models():
    """Get available models"""
    try:
        api_key = get_api_key()
        if not api_key:
            return jsonify({'error': 'API key not found'}), 400

        client = genai.Client(api_key=api_key)
        
        # For now, return the known models
        models = [
            {
                'name': 'gemini-2.5-flash-image-preview',
                'displayName': 'Gemini 2.5 Flash Image Preview',
                'description': 'Latest Gemini model with image processing capabilities'
            }
        ]
        
        return jsonify({'models': models})
    
    except Exception as e:
        return jsonify({'error': f'Failed to fetch models: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded/generated files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/test')
def test_api():
    """Test API connection"""
    try:
        api_key = get_api_key()
        if not api_key:
            return jsonify({'status': 'error', 'message': 'API key not found'}), 400
        
        client = genai.Client(api_key=api_key)
        
        # Try to list models first (lighter operation)
        try:
            models = list(client.models.list())
            if models:
                return jsonify({
                    'status': 'success',
                    'message': 'API connection successful (key valid)',
                    'available_models': len(models)
                })
        except Exception as list_error:
            # If listing fails, try a minimal generation call
            pass
        
        # Fallback: minimal test call
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text="Hi")]
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[content],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=10
            )
        )
        
        return jsonify({
            'status': 'success',
            'message': 'API connection successful',
            'test_response': response.text if hasattr(response, 'text') else str(response)
        })
    
    except Exception as e:
        error_str = str(e)
        if '429' in error_str and 'RESOURCE_EXHAUSTED' in error_str:
            return jsonify({
                'status': 'quota_exceeded',
                'message': 'API密钥有效，但免费配额已用完。请等待配额重置或升级到付费计划。',
                'error_type': 'quota_limit'
            }), 200  # 返回200因为密钥是有效的
        elif '401' in error_str or 'UNAUTHENTICATED' in error_str:
            return jsonify({
                'status': 'error',
                'message': 'API密钥无效或已过期',
                'error_type': 'auth_error'
            }), 400
        else:
            return jsonify({
                'status': 'error',
                'message': f'连接测试失败: {error_str}',
                'error_type': 'unknown'
            }), 500

@app.route('/download/<filename>')
def download_image(filename):
    """Download image with format conversion"""
    try:
        format_type = request.args.get('format', 'png').lower()
        quality = int(request.args.get('quality', 90))
        
        # Security check
        filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
        
        # Open and convert image
        with Image.open(filepath) as img:
            # Convert to RGB if necessary (for JPEG)
            if format_type == 'jpg' and img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create output buffer
            output = io.BytesIO()
            
            # Save with specified format
            if format_type == 'png':
                img.save(output, format='PNG', optimize=True)
                mimetype = 'image/png'
                ext = 'png'
            elif format_type == 'jpg':
                img.save(output, format='JPEG', quality=quality, optimize=True)
                mimetype = 'image/jpeg'
                ext = 'jpg'
            elif format_type == 'webp':
                img.save(output, format='WEBP', quality=quality, optimize=True)
                mimetype = 'image/webp'
                ext = 'webp'
            else:
                return jsonify({'error': '不支持的格式'}), 400
            
            output.seek(0)
            
            # Generate download filename
            base_name = os.path.splitext(filename)[0]
            download_filename = f"{base_name}.{ext}"
            
            return send_file(
                output,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_filename
            )
    
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

def get_all_local_ips():
    """Get all local IP addresses"""
    ips = []
    try:
        # Get network configuration on Windows
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        
        # Try multiple patterns to match different Windows versions
        patterns = [
            r'IPv4.*?[：:]\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # Chinese version
            r'IPv4 Address.*?[：:]\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # English version
            r'IP Address.*?[：:]\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # Alternative format
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, result.stdout, re.IGNORECASE)
            for ip in matches:
                # Skip localhost and link-local addresses
                if not ip.startswith('127.') and not ip.startswith('169.254.') and ip not in ips:
                    ips.append(ip)
                    
        # If no matches found, try a simpler approach
        if not ips:
            # Look for any IP pattern in the output
            simple_pattern = r'([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)'
            all_matches = re.findall(simple_pattern, result.stdout)
            for ip in all_matches:
                if (not ip.startswith('127.') and 
                    not ip.startswith('169.254.') and 
                    not ip.startswith('255.') and
                    ip not in ips):
                    ips.append(ip)
                    
    except Exception as e:
        print(f"Error getting IPs: {e}")
        pass
    
    return ips

def start_ssh_tunnel():
    """Start SSH tunnel and return public URL"""
    print("\n🚀 正在启动SSH隧道...")
    
    tunnel_services = [
        "serveo.net",
        "ssh.localhost.run"
    ]
    
    for service in tunnel_services:
        try:
            print(f"正在尝试连接 {service}...")
            
            # Start SSH tunnel process
            cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -R 80:localhost:5010 {service}"
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for tunnel to establish and get URL
            start_time = time.time()
            while time.time() - start_time < 15:  # 15 second timeout
                if process.poll() is not None:
                    break
                    
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                    
                    # Look for public URL patterns
                    url_patterns = [
                        r'https?://[\w\-\.]+\.(serveo\.net|localhost\.run)',
                        r'Forwarding HTTP traffic from (https?://[\w\-\.]+)'
                    ]
                    
                    for pattern in url_patterns:
                        match = re.search(pattern, output)
                        if match:
                            public_url = match.group(1) if match.lastindex else match.group(0)
                            if public_url.startswith('http'):
                                print(f"\n🎉 隧道建立成功！")
                                print(f"📱 公网地址: {public_url}")
                                print(f"🔗 分享此链接给其他用户")
                                return process, public_url
                
                time.sleep(1)
            
            # If we get here, tunnel failed
            process.terminate()
            print(f"❌ 连接 {service} 失败")
            
        except Exception as e:
            print(f"❌ 连接 {service} 出错: {e}")
    
    print("\n⚠️  SSH隧道服务暂时不可用")
    print("\n🌐 替代方案:")
    
    # Show available network IPs
    all_ips = get_all_local_ips()
    if all_ips:
        for ip in all_ips:
            if ip.startswith('10.'):
                print(f"   1. 局域网访问: http://{ip}:5010")
            elif ip.startswith('26.'):
                print(f"   2. VPN网络访问: http://{ip}:5010")
    
    print("   3. 手动启动ngrok: ngrok http 5010")
    print("   4. 使用其他内网穿透工具")
    return None, None

def start_tunnel_thread():
    """Start tunnel in background thread"""
    def tunnel_worker():
        time.sleep(2)  # Wait for Flask to start
        process, url = start_ssh_tunnel()
        if process:
            # Keep tunnel alive
            try:
                process.wait()
            except:
                pass
    
    tunnel_thread = threading.Thread(target=tunnel_worker, daemon=True)
    tunnel_thread.start()

if __name__ == '__main__':
    print("="*60)
    print("🎨 Gemini 2.5 Flash 图像生成服务")
    print("="*60)
    print(f"📁 上传目录: {app.config['UPLOAD_FOLDER']}")
    print("\n📍 访问地址:")
    print(f"   本地: http://localhost:5010")
    
    # Get all network IPs
    all_ips = get_all_local_ips()
    if all_ips:
        for ip in all_ips:
            if ip.startswith('10.'):
                print(f"   局域网: http://{ip}:5010")
            elif ip.startswith('26.'):
                print(f"   VPN网络: http://{ip}:5010")
            else:
                print(f"   网络: http://{ip}:5010")
    
    print("\n🌐 正在启动公网隧道...")
    
    # Start tunnel in background
    start_tunnel_thread()
    
    print("\n🚀 启动Flask服务器...")
    print("按 Ctrl+C 停止服务")
    print("="*60)
    
    try:
        # Run the app accessible from network
        app.run(host='0.0.0.0', port=5010, debug=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
