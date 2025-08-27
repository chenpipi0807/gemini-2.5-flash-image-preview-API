import os
import base64
import mimetypes
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import mimetypes
from datetime import datetime
import google.genai as genai
from google.genai import types
import base64

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

        # Get form data
        text_prompt = request.form.get('text_prompt', '')
        model_name = request.form.get('model', 'gemini-2.5-flash-image-preview')
        temperature = float(request.form.get('temperature', 1.0))
        max_output_tokens = int(request.form.get('max_output_tokens', 8192))
        top_p = float(request.form.get('top_p', 0.95))
        top_k = int(request.form.get('top_k', 40))
        
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
        if text_prompt:
            parts.append(types.Part.from_text(text=text_prompt))

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

        # Build generation config
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_output_tokens,
            'top_p': top_p,
            'top_k': top_k
        }

        # Check if image generation is requested
        response_modalities = request.form.get('response_modalities', 'TEXT')
        if response_modalities == 'TEXT,IMAGE':
            modalities = ['TEXT', 'IMAGE']
        else:
            modalities = ['TEXT']
        
        # Make API call
        response = client.models.generate_content(
            model=model_name,
            contents=[content],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
                safety_settings=safety_settings_list,
                response_modalities=modalities
            )
        )

        # Process response - support both text and image outputs
        response_text = ""
        generated_images = []
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for i, part in enumerate(candidate.content.parts):
                    # Handle text parts
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
                    
                    # Handle image parts (inline_data)
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        # Save binary image data
                        image_data = part.inline_data.data
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
                        filename = f'generated_image_{timestamp}_{i}.png'
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(base64.b64decode(image_data))
                        
                        generated_images.append(filename)
                        
                        # Add image info to response text
                        if not response_text:
                            response_text = f"✨ 已生成图像: {filename}"
                        else:
                            response_text += f"\n\n✨ 已生成图像: {filename}"
        
        # Fallback to old method if no parts found
        if not response_text and not generated_images:
            response_text = response.text if hasattr(response, 'text') else str(response)
        
        result = {
            'success': True,
            'response': response_text,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': generation_config,
            'generated_images': generated_images
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'API call failed: {str(e)}'}), 500

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)
