# Machine Learning Engineer Nanodegree

## Capstone Project

Tahsin Mayeesha

Date :

## I. Definition

(approx. 1-2 pages

Fishery data

### Project Overview

In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:

-   Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?
-   Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?

https://www.engadget.com/2016/11/18/how-humans-and-machines-can-save-the-tuna

https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

https://www.theguardian.com/sustainable-business/2016/nov/20/artificial-intelligence-illegal-fishing-tuna-sharks

http://fortune.com/2016/11/14/deep-learning-artificial-intelligence-tuna-industry/?iid=leftrail

### Problem Statement

In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:

-   Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?

-   Have you thoroughly discussed how you will attempt to solve the problem?

-   Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?

      ​

Identifying which species of fishes are being caught from raw video footage is a time consuming and expensive task and there's often too much information to comprehend for a single human observer. Many species of fishes look fairly similar from a camera and often there's a lot of activity going on in a fishing boat during work which makes it hard to understand what's going on. On top of that fishers involved in illegal fishing often has incentives to create more distraction to hide their crimes. Deep learning techniques for image classification can help by automatically classifying species of tunas, sharks and more that fishing boats catch, which will accelerate the video review process. Faster review and more reliable data will enable countries to reallocate human capital to enforce laws and understand which species are being caught most often so that ocean management can be more effective. 

In this competition, [The Nature Conservancy](http://www.thisisourfuture.org/) asks to detect which species of fish appears on a fishing boat, using based on images captured from boat cameras of various angles.  The goal is to predict the likelihood that a fish is from a certain class from the provided classes, thus making it a multi-class classification problem in machine learning terms. Eight target classes are provided in this dataset : Albacore tuna, Bigeye tuna, Yellowfin tuna, Mahi Mahi, Opah, Sharks, Other (meaning that there are fish present but not in the above categories), and No Fish (meaning that no fish is in the picture). Each image has only one fish category, except that there are sometimes very small fish in the pictures that are used as bait. The Nature Conservancy also has kindly provided a visualization of labels, as the raw images can be triggering for many people.

![Images of all fish labels](https://kaggle2.blob.core.windows.net/competitions/kaggle/5568/media/species-ref-key.jpg)

As deep learning techniques have been very effective in image classification over the years, in this project, transfer learning along with data augmentation will be used to train a convolutional neural network to classify images of fish to their respective classes. 



### Metrics



In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:

-   Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?

-   Have you provided reasonable justification for the metrics chosen based on the problem and solution?

The metric for this competition is multi-class log loss (also known as categorical cross entropy)


$$
log loss = \frac{1}{N}\sum_{i}^{N}\sum_{j}^{M}y_{ij}\log(p_{ij})
$$
where N is the number of images in the test set, M is the number of image class labels







## II. Analysis

(approx. 2-4 pages)

### Data Exploration

In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:

-   If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?

-   If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?

-   If a dataset is not present for this problem, has discussion been made about the input space or input data for your problem?

-   Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)

### Exploratory Visualization

In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:

-   Have you visualized a relevant characteristic or feature about the dataset or input data?

-   Is the visualization thoroughly analyzed and discussed?

-   If a plot is provided, are the axes, title, and datum clearly defined?

### Algorithms and Techniques

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:

-   Are the algorithms you will use, including any default variables/parameters in the project clearly defined?

-   Are the techniques to be used thoroughly discussed and justified?

-   Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?

### Benchmark

In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:

-   Has some result or value been provided that acts as a benchmark for measuring performance?

-   Is it clear how this result or value was obtained (whether by data or by hypothesis)?

My writing :

-   Random baseline :

-   “Smarter baseline” :

## 

## III. Methodology

(approx. 3-5 pages)

-   K-nearest neighbors approach

-   Convolutional neural network from scratch approach

-   Transfer Learning Approach

### Data Preprocessing

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:

-   If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?

-   Based on the Data Exploration section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?

-   If no preprocessing is needed, has it been made clear why?

### Implementation

In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:

-   Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?

-   Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?

-   Was there any part of the coding process (e.g., writing complicated functions) that should be documented?

### Refinement

In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:

-   Has an initial solution been found and clearly reported?

-   Is the process of improvement clearly documented, such as what techniques were used?

-   Are intermediate and final solutions clearly reported as the process is improved?

## IV. Results

(approx. 2-3 pages)

### Model Evaluation and Validation

In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:

-   Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?

-   Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?

-   Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?

-   Can results found from the model be trusted?

### Justification

In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:

-   Are the final results found stronger than the benchmark result reported earlier?

-   Have you thoroughly analyzed and discussed the final solution?

-   Is the final solution significant enough to have solved the problem?

## V. Conclusion

(approx. 1-2 pages)

### Free-Form Visualization

In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:

-   Have you visualized a relevant or important quality about the problem, dataset, input data, or results?

-   Is the visualization thoroughly analyzed and discussed?

-   If a plot is provided, are the axes, title, and datum clearly defined?

### Reflection

In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:

-   Have you thoroughly summarized the entire process you used for this project?

-   Were there any interesting aspects of the project?

-   Were there any difficult aspects of the project?

-   Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?

### Improvement

In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:

-   Are there further improvements that could be made on the algorithms or techniques you used in this project?

-   Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?

-   If you used your final solution as the new benchmark, do you think an even better solution exists?

Before submitting, ask yourself. . .

-   Does the project report you’ve written follow a well-organized structure similar to that of the project template?

-   Is each section (particularly Analysis and Methodology) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?

-   Would the intended audience of your project be able to understand your analysis, methods, and results?

-   Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?

-   Are all the resources used for this project correctly cited and referenced?

-   Is the code that implements your solution easily readable and properly commented?

-   Does the code execute without error and produce results similar to those reported?


