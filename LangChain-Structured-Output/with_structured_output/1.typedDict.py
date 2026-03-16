from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct",
    task = "text_generation"
)

model = ChatHuggingFace(llm=llm)
 
# #schema
# class Review(TypedDict):
#     summary:str
#     sentiment:str

# structured_model = model.with_structured_output(Review)


# result = structured_model.invoke('The display is bright and vibrant, making it great for watching videos and browsing the web. I also like the camera quality — photos come out sharp and detailed in both daylight and low-light conditions.')


# print(result)

# print(result['summary'])
# print(result['sentiment'])




#annotated-typedict
class Review(TypedDict):
    keyThemes:Annotated[str,"Note down the key specifications or themes from the review"]
    summary:Annotated[str,'A brief intro of what the review means to tell']
    sentiment:Annotated[Literal['pos','neg'],"Return sentiment of review either positive negative or neutral"]
    pros:Annotated[Optional[list[str]],'Write down all the pros inside list as string']
    cons:Annotated[Optional[list[str]],'Write down all the cons inside list as string']
    name:Annotated[Optional[str],"Name of the reviewer. Return null if not mentioned in the review"]

structured_model = model.with_structured_output(Review)


result = structured_model.invoke('I have been using this smartphone for about three weeks now, and overall it has been a very impressive device, especially considering its price range. The design is sleek and premium, with a sturdy aluminum frame and a matte glass back that feels comfortable to hold and resists fingerprints fairly well.One of the standout aspects of this phone is its display. The 6.7-inch OLED screen is bright, vibrant, and extremely sharp. Colors appear accurate, and the 120Hz refresh rate makes scrolling and animations feel very smooth. Watching videos and playing games on this display is a great experience, even under bright outdoor lighting.The performance of the device is also excellent. Powered by a modern processor with 12GB of RAM, the phone handles multitasking, gaming, and everyday apps without any noticeable lag. I regularly switch between multiple applications, and everything remains responsive and fluid.The camera system is another highlight. The main 50MP sensor captures sharp and detailed photos with good dynamic range. Portrait mode works well, and the ultrawide lens is useful for landscapes and group photos. Night mode significantly improves low-light photography, although it sometimes takes a second longer to process images.Battery life is solid as well. The 5000mAh battery comfortably lasts a full day of moderate to heavy use, including browsing, social media, and video streaming. Fast charging is also convenient, allowing the phone to reach about 70percent charge in around 30 minutes.Overall, the device offers a balanced combination of performance, display quality, and battery life, making it a strong option for users who want a premium experience without paying flagship prices.')

print(result['sentiment'])
print(result)





    



