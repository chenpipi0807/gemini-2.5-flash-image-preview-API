import os
import base64
import mimetypes
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import google.genai as genai
from google.genai import types
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_content():
    """Main API endpoint for generating content with Gemini"""
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
        
        # Safety settings - comprehensive coverage
        safety_settings_list = [
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold=request.form.get('safety_harm_category_harassment', 'BLOCK_MEDIUM_AND_ABOVE')
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold=request.form.get('safety_harm_category_hate_speech', 'BLOCK_MEDIUM_AND_ABOVE')
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold=request.form.get('safety_harm_category_sexually_explicit', 'BLOCK_MEDIUM_AND_ABOVE')
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold=request.form.get('safety_harm_category_dangerous_content', 'BLOCK_MEDIUM_AND_ABOVE')
            )
        ]

        # Initialize client
        client = genai.Client(api_key=api_key)

        # Build content parts
        parts = []
        
        # Add text part if provided
        if prompt:
            parts.append(types.Part.from_text(text=prompt))

        # Handle uploaded images
        uploaded_files = request.files.getlist('images')
        for file in uploaded_files:
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
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

        # Build generation config with seed support (like ComfyUI)
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_output_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'seed': seed
        }

        # 根据用户选择和上传文件确定响应模式
        response_modalities = request.form.get('response_modalities', 'TEXT,IMAGE')
        
        if not uploaded_files:
            print("🎨 检测到文本生图模式，强制启用图像生成")
            modalities = ["TEXT", "IMAGE"]  # 文生图模式
        else:
            # 有上传图片时，根据用户选择决定是否生成图像
            if response_modalities == 'TEXT,IMAGE':
                print("🖼️ 检测到图像编辑模式，启用图像生成")
                modalities = ["TEXT", "IMAGE"]  # 图像编辑模式
            else:
                print("🔍 检测到图像分析模式，仅文本输出")
                modalities = ["TEXT"]  # 纯分析模式
        
        # Debug: Print API call parameters
        print(f"🚀 API调用参数:")
        print(f"  - 模型: {model_name}")
        print(f"  - 响应模式: {modalities}")
        print(f"  - 提示词: {prompt[:100] if prompt else 'None'}...")
        print(f"  - 上传文件数: {len([f for f in uploaded_files if f and f.filename != ''])}")
        print(f"  - 随机种子: {seed}")
        print(f"  - 温度: {temperature}")
        print(f"  - 最大输出: {max_output_tokens}")
        
        # Make API call using streaming (required for image generation)
        stream = client.models.generate_content_stream(
            model=model_name,
            contents=[content],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                safety_settings=safety_settings_list,
                response_modalities=modalities
            )
        )

        # Process streaming response - support both text and image outputs
        response_text = ""
        generated_images = []
        file_index = 0
        
        print(f"🔍 开始处理流式响应...")
        
        for chunk in stream:
            print(f"📦 收到chunk: {type(chunk)}")
            
            # Skip empty chunks
            if (chunk.candidates is None or 
                chunk.candidates[0].content is None or 
                chunk.candidates[0].content.parts is None):
                print("  - 跳过空chunk")
                continue
            
            for i, part in enumerate(chunk.candidates[0].content.parts):
                print(f"  - Part {i}: {type(part)}")
                
                # Handle image parts first (like official example)
                if (hasattr(part, 'inline_data') and part.inline_data and 
                    hasattr(part.inline_data, 'data') and part.inline_data.data):
                    try:
                        print(f"🎨 发现图像数据！")
                        
                        # Save image like official example
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        filename = f'generated_image_{timestamp}_{file_index}.png'
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        inline_data = part.inline_data
                        data_buffer = inline_data.data
                        
                        # Get file extension from mime type
                        if hasattr(inline_data, 'mime_type') and inline_data.mime_type:
                            file_extension = mimetypes.guess_extension(inline_data.mime_type)
                            if file_extension:
                                filename = f'generated_image_{timestamp}_{file_index}{file_extension}'
                                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        # Save binary data directly
                        with open(filepath, 'wb') as f:
                            f.write(data_buffer)
                        
                        file_size = os.path.getsize(filepath)
                        print(f"✅ 图片已保存: {filepath} ({file_size} 字节)")
                        
                        if file_size > 100:  # Valid file
                            generated_images.append(filename)
                            file_index += 1
                        else:
                            print(f"❌ 文件太小，删除: {filepath}")
                            os.remove(filepath)
                            
                    except Exception as img_error:
                        print(f"❌ 图片保存失败: {img_error}")
                
                # Handle text parts
                elif hasattr(part, 'text') and part.text:
                    print(f"📝 收到文本: {part.text[:50]}...")
                    response_text += part.text
                
                # Also handle chunk.text for streaming text (fallback)
                elif hasattr(chunk, 'text') and chunk.text:
                    print(f"📝 收到流式文本: {chunk.text[:50]}...")
                    response_text += chunk.text
        
        # Debug: Print response structure
        print(f"🔍 调试信息:")
        print(f"  - 响应文本长度: {len(response_text)}")
        print(f"  - 生成图片数量: {len(generated_images)}")
        print(f"  - 图片文件名: {generated_images}")
        
        # Fallback to old method if no parts found
        if not response_text and not generated_images:
            response_text = response.text if hasattr(response, 'text') else str(response)
            print(f"  - 使用备用方法获取响应: {response_text[:100]}...")
        
        # Convert local filenames to full URLs
        base_url = request.url_root.rstrip('/')
        generated_image_urls = [f"{base_url}/uploads/{filename}" for filename in generated_images]
        
        result = {
            'success': True,
            'response': response_text,
            'generated_images': generated_images,
            'generated_image_urls': generated_image_urls,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'temperature': temperature,
                'max_output_tokens': max_output_tokens,
                'top_p': top_p,
                'top_k': top_k
            }
        }
        
        print(f"📤 返回结果: {len(generated_images)} 张图片")

        return jsonify(result)

    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return jsonify({'error': f'API调用失败: {str(e)}'}), 500

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
