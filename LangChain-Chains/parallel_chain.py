from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
)
model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

)
model2 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template = 'Generate me a short and simple note from the following text \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short questions and asnwers from the following text \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n {notes} and {quiz}',
    input_variables = ['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 |model2 | parser
    }
)

merge_chain = prompt3 | model2 | parser 

chain = parallel_chain | merge_chain

text = """Study Guide: Support Vector Machines (SVM)
1. Overview

Support Vector Machines (SVMs) are supervised machine learning models used for:

Classification
Regression
Outlier detection

They work by finding an optimal hyperplane that separates data classes with the maximum margin.

2. Advantages of SVM
Works well in high-dimensional spaces
Effective even when number of features > number of samples
Uses only support vectors → memory efficient
Flexible due to different kernel functions (linear, RBF, polynomial, etc.)
3. Disadvantages of SVM
Computationally expensive for large datasets
Sensitive to parameter tuning (C, kernel, gamma)
Does not directly provide probability estimates
Performance degrades if features are not properly scaled
4. Core Concept

SVM finds a decision boundary (hyperplane) that maximizes the margin between classes.

Only a subset of training points, called support vectors, influence the final model.

5. Types of SVM
5.1 Classification
SVC (Support Vector Classifier)
NuSVC
LinearSVC (faster linear version)

These models can handle:

Binary classification
Multi-class classification
5.2 Regression
SVR (Support Vector Regression)
NuSVR
LinearSVR

Used when output is continuous rather than categorical.

5.3 Outlier Detection
OneClassSVM is used for novelty and anomaly detection.
6. Kernel Functions

SVM can transform data using kernel tricks:

Linear: simple dot product
Polynomial: captures polynomial relationships
RBF (Gaussian): handles non-linear patterns
Sigmoid: similar to neural activation
7. Important Parameters
C: controls regularization (lower = more regularization)
Kernel: defines transformation type
Gamma: controls influence of single training points
Class weight: handles imbalanced datasets
8. Practical Considerations
Always scale features before training
SVM is sensitive to feature magnitude
Use pipelines for preprocessing + model training
Large datasets require careful optimization or linear SVM
9. Key Insight

SVM does not try to separate all points perfectly.
Instead, it finds the best margin that generalizes well to unseen data.

10. Summary

SVM is a powerful, mathematically grounded algorithm that:

Works well for complex datasets
Depends heavily on support vectors
Requires proper tuning and preprocessing

It remains one of the most important classical machine learning algorithms."""
result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()