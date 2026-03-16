# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

# LangChain HuggingFace ChatHuggingFace currently does NOT support using Pydantic schemas directly for structured output. Right now, the with_structured_output() method in ChatHuggingFace only works with TypedDict or structured output via StructuredOutputParser, not Pydantic.

from langchain_groq import ChatGroq


from dotenv import load_dotenv
from pydantic import BaseModel , Field 
from typing import Optional , Literal


load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = "Qwen/Qwen2.5-72B-Instruct",
#     task = "text_generation"
# )

# model = ChatHuggingFace(llm=llm)


# 3️⃣ Initialize the Groq Chat model
groq_model = ChatGroq(
    model="llama-3.3-70b-versatile",  # ✅ updated model name
    temperature=0,
    max_tokens=512
)

#schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

structured_model = groq_model.with_structured_output(json_schema)


result = structured_model.invoke('I have been using this smartphone for about three weeks now, and overall it has been a very impressive device, especially considering its price range. The design is sleek and premium, with a sturdy aluminum frame and a matte glass back that feels comfortable to hold and resists fingerprints fairly well.One of the standout aspects of this phone is its display. The 6.7-inch OLED screen is bright, vibrant, and extremely sharp. Colors appear accurate, and the 120Hz refresh rate makes scrolling and animations feel very smooth. Watching videos and playing games on this display is a great experience, even under bright outdoor lighting.The performance of the device is also excellent. Powered by a modern processor with 12GB of RAM, the phone handles multitasking, gaming, and everyday apps without any noticeable lag. I regularly switch between multiple applications, and everything remains responsive and fluid.The camera system is another highlight. The main 50MP sensor captures sharp and detailed photos with good dynamic range. Portrait mode works well, and the ultrawide lens is useful for landscapes and group photos. Night mode significantly improves low-light photography, although it sometimes takes a second longer to process images.Battery life is solid as well. The 5000mAh battery comfortably lasts a full day of moderate to heavy use, including browsing, social media, and video streaming. Fast charging is also convenient, allowing the phone to reach about 70percent charge in around 30 minutes.Overall, the device offers a balanced combination of performance, display quality, and battery life, making it a strong option for users who want a premium experience without paying flagship prices.')


print(result)





    



