# Task 3
## Summary of Task 1
Task one establishes the basic sentence transformer architecture and roughs out the code to train the model contrastively.  The decision was made to use the similarity loss in a constrastive setting because of the downstream tasks selected.  As a form of metric learning, we can use the similarity loss to maximize the distance between embeddings for disimilar sentences and minimize the disntance between embeddings for similar sentences.  This will give greater fleximbilty when optimizing for downstream tasks because the classification heads will have an easier job drawing the class boundaries in latent space.  

Archtectural choices which were made: I chose to begin with a pretrained BERT encoder which already performs well on a variety of benchmarks.  This gives us a good initialization to then perform the contrastive finetune.  With the goal of updating the model backbone, this step is performed with all weights unfrozen.  

## Summary of Task 2
Task two aims to build a multi-task learner on top of the large backbone we finetuned in task one.  The two tasks included here are sentence classification, in which a supervised dataset contains sentences and their assigned class.  To accomplish this task, I augmented the model with a single layer classification head on top of the transformer backbone.  The classification head takes the mean pooled, last hidden state as input and outputs the logits for each predetermined class.  

The second task is a sentiment analysis.  Similar to the classification task, here I attach a single layer on top of the transformer backbone which consumes the mean pooled, last hidden state, of the transformer and outputs a scalar value.  The scalar value may be passed through a sigmoid to attain a sentiment between 0 and 1.  

In both of these tasks, during the training process, the transformer backbone weights should be frozen such that we are only optimizing over the output heads (classification and sentiment) individually.  The training for these tasks should be performed using separate supervised datasets and the the individual heads should be trained independently.  

The benefits of training with the frozen backbone are twofold: first this maintains the generalizability of our foundation model in the case that future downstream tasks should arise.  Secondly, it is cost effective in that we save on compute and time (and thus money) by only training the individual heads.  For any downstream task we can now pass our input dataset through the transforme a single time, save the embeddings do disk, and then perform a quick efficient supervised training on just the task heads. 

# Task 4
My two hours is up, but if I am selected for the next steps I would love to discuss the ideas behind a layer-wise lr :).   Thank you for considering my submission!
