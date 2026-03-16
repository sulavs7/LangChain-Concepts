from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str = "sulav"#default args
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0 , lt = 10)
    # cgpa : float = Field(gt=0 , lt = 10 , default=5, description = "a decimal value representing cgpa of student")# default value and description

# new_student = {'name':'sulav'}
# new_student = {'name':32} # gives error "Input should be a valid string"
# new_student = {}
# new_student = {'age':32}
# new_student = {'name':'rajesh','age':32 }

# new_student = {'name':'rajesh','age':'32' ,'email':'abc'}#error :value is not a valid email address
# new_student = {'name':'rajesh','age':'32' ,'email':'abc@gmail.com'}

# new_student = {'name':'rajesh','age':'32' ,'email':'abc@gmail.com','cgpa' : 15} #error :Input should be less than 10
new_student = {'name':'rajesh','age':'32' ,'email':'abc@gmail.com','cgpa' : 5} 



student = Student(**new_student) # ** spreads the dictionary so Student(name="sulav")

print(student)

student_dict = dict(student)
print(student_dict['age'])