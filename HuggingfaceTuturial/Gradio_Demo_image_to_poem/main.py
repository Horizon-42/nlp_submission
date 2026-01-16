import gradio as gr
import google.generativeai as genai
import base64
from io import BytesIO
from PIL import Image

# Set up Google GenAI API Key (Replace with your actual API key)
genai.configure(api_key="APIKey")  

def image_to_poem(image,language):
    """Generates a poem about an upload image using Gemini 2.5 flash"""

    try:
        # Convert image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        # Convert image to Base64 for Gemini API
        image_b64 = base64.b64encode(img_bytes).decode('utf-8')
        prompt = "Describe this image in one sentence."   #Question to ask Gemini about the input image 

        # Use Gemini 2.5 flash for image analysis
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([{"mime_type": "image/png", "data": image_b64}, prompt]) #get the response here 
        description = response.text if response else "No description available."

        # Generate poem based on the description
        response_poem = model.generate_content(f"Based on this image description: {description}, write a short poem in {language}.") # ask Gemini again to generate a short poem based on the description
        generated_poem = response_poem.text if response_poem else "Could not generate a poem."


        return generated_poem

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Define Gradio Interface
interface = gr.Interface(
    fn=image_to_poem,
    inputs= [gr.Image(type="pil", label="Upload an image"),gr.Textbox(label= "Language", placeholder="English / German / Hindi /etc.", info="Enter the language you want to generate a poem in.")],
    outputs=gr.Textbox(label="Generated Poem",lines=5, max_length=10),
    title="AI Poem Generator",
    description="Upload an image, enter the language, and AI will generate a poem about it."
)

if __name__ == "__main__":
    interface.launch()
