# AirBnb-Reviews

Model Training
Train the model:
The model is trained using the DistilBERT architecture with the following configuration:
Epochs: 1
Batch Size: 128
Learning Rate: Optimized with weight decay
Save the model:
After training, the model and tokenizer are saved to the directory ./fine-tuned-distilbert.
Evaluation
Evaluate the model:
The model's performance is evaluated on the test set using accuracy and F1 score. Metrics and confusion matrix are printed to the console.
To evaluate the model, run:
Visualization
Confusion Matrix:
The confusion matrix is plotted and saved as an image file to visualize the model's classification performance.
To generate and view the confusion matrix, run:
Dependencies
The following libraries are required to run the code:
pandas ==2.2.2
datasets==2.20.0
transformers==4.42.3
scikit-learn==1.2.2
matplotlib==3.7.5
seaborn==0.12.2
geopandas==0.14.4
folium==0.17.0
