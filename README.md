[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ytAIXJ0n)
## Mudd 2nd Floor Rare Class Classification of Images

Please read the entirety of this README file, which contains description of the instructions on completing the assignment and submission details. You will need to modify this file before submission.

### <span style="color:red">**[Submission Deadline]:**</span> 11:59 PM, April 4th, 2025

## Introduction
In this assignment, you will work with a dataset of images of street traffic captured at the 120th Amsterdam Ave. and 120th Street intersection. You are asked to come up with methods to identify image classes and situations that have potential to disrupt traffic flow or cause harm to pedestrians. 

You need to experiment with the dataset extensively and explore advanced deep learning methods and techniques. In particular, you will work with a "multi-label" classification task (also known as "image tagging") on a dataset more complex than your typical Kaggle or benchmark (such as Pascal-VOC and ImageNet) datasets. 

You will have to explore more advanced methods than those typically taught in an introductory deep learning course for solving this assignment. You will need to spend some time searching for, testing out, or coming up with your own deep learning methods.

<font color="red"><strong>NOTE:</strong></font> This dataset can be challenging to work with. Please start early - you will not be able to finish the assignment in 5-6 days only.


## <span style="color:red"><strong>TODO:</strong></span> Group Member Names and UNIs
**This assignment should be done in groups of 2-3 members**. Please list the names and UNIs of the group members below:

- Arsalan Firoozi: af3410
- Riki Shimizu: rs4613
- Kabir Bagai: kb3343

## <span style="color:red"><strong>TODO:</strong></span> (Re)naming of Repository

***INSTRUCTIONS*** for naming the students' solution repository for this group assignment, please add the UNIs for all group members separated with a dash "-" using the following template:
* Template: `e6691-2025spring-assign2-UNI1-UNI2-UNI3`
* Example: `e6691-2025spring-assign2-zz9999-aa9999-aa0000`
* Bad example: `e6691-2025spring-assign2-2ndFloor-zz9999-aa9999-aa0000`
* Bad example: `e6691-2025spring-assign2-e6691-2025spring-assign2-zz9999-aa9999-aa0000`
* This can be done from the "Settings" on the repository webpage.

## <font color="red"><strong>NOTE:</strong></font> Before you start
1. Set up your computing environment (preferably GCP, since you will have to deal with a fairly large dataset, and use state-of-the-art models if you want good results). Install NVIDIA GPU Drivers, CUDA, Jupyter, etc. Refer to assignment 1 for instructions on how to set up a GCP VM instance for machine learning/deep learning development.
2. When selecting/using and/or developing one (or more) deep learning methods to perform predictions on your dataset, you are allowed to select any model of your choice. However, keep the following in mind:
    * <span style="color:red">***You must clearly reference  the literature and/or external code implementations that you have used in your code***</span>.
    * If you use externally available code, you must add comments into the code that describe in layman's words what the model and its functions execute
    * You must use PyTorch as the framework of choice.
    * <span style="color:red">***You must include your code and comments/descriptions in the designated parts of the `.ipynb` notebook that demonstrates how you have developed your model*** </span>.




 
## Detailed instructions on how to submit your solution to this assignment/homework

### <span style="color:red">**[Submission Deadline]:**</span> 11:59 PM, April 4th, 2025

1. Include all your code in the GitHub repository before submission. **Do not include any data or labels in the GitHub**.

2. (You can skip this step if you only run inference on the models, and no training) If you train any models include model weights in a Google drive folder and include a link in the **Model Google Drive Link** section below and make sure to share the drive with the TA team (e6040tas@columbia.edu). 

3. The solution to the assignment must be pushed into your repository.

4. Please make sure that all your code is working and runnable.

5. Additionally, **export a PDF version** of your notebook and also include it in your repository.

6. Add your report (.pdf file) to the GitHub repository before submission.

7. **Make sure your changes are properly pushed to Github, including the exported PDF files**. Always open the repository webpage and check if you could see your modifications. **We will not do any regrades for incomplete submissions beyond what you have submitted. Furthermore, we will only grade based on your latest submission to Gradescope. Any submission before the latest one will not be considered.**

8. Submit your final repository to Gradescope (can be found from Courseworks). Students will need to enter Gradescope and link the GitHub repo for submission. <span style="color:red">***Each group member needs to submit the GitHub link in Gradescope***</span> even though it is a group assignment. Always open the "Codes" tab in Gradescope and check the completeness of the contents after submission.

## Group Late Days
Since this assignment is to be done in groups, the late day policy is slightly adjusted. The late deadline set for this whole assignment is 4 days after the original assignment deadline, so any submissions after April 4th 11:59 PM untill April 8th 11:59 PM will be accounted into the **late days that for your whole team (which means the late days will be counted towards every student in your team)**, so coordinate with your teammate to make sure no one exhausts their late days.


## Model Google Drive Link:

We did not train any new model in this. We used a model called Recognize Anything, which can be accessed at: https://github.com/xinyu1205/recognize-anything.


## Grading

- Setup Data <span style="color:red">(10%)</span>
- Complete Evaluation Code <span style="color:red">(5%)</span>
- Explore and Build Your Own Model <span style="color:red">(30%)</span>
- Evaluate your method on the training set <span style="color:red">(5%)</span>
- Prepare for Kaggle <span style="color:red">(10%)</span>
- Evaluate your model (Kaggle) <span style="color:red">(10%)</span>
- Write a report <span style="color:red">(20%)</span>
- Final Submission and repo re-naming, at least 3 commits per student <span style="color:red">(10%)</span>


