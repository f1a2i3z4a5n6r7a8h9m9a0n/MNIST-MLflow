# MNIST experiment results



Tracking mode: local. Local runs are not DagsHub evidence.



Selected final-epoch model: **experiment2_256_adam_b64**, using validation accuracy first and lower validation loss to break ties.

Validation accuracy: 0.982000; validation loss: 0.137461.

The next-ranked model, baseline_128_adam_b64, has validation accuracy 0.980000 and validation loss 0.134100. These observed validation values determine the ranking, rather than model size or training accuracy.

Training minus validation accuracy: 0.018000. Minimum validation loss occurred at epoch 6.



Final validation loss is 0.062214 above its minimum. Since that epoch, training loss changed by -0.030027. Validation loss increased while training loss decreased, suggesting overfitting after that epoch.

A training/validation gap, especially when training loss falls while validation loss rises, is evidence of possible overfitting. The curves describe this sample; one seed does not establish that an architecture is always superior. Training metrics are the last fit epoch's batch aggregates; validation metrics evaluate the final epoch weights.



All models use final epoch weights; there is no early stopping or best-epoch restoration. The test dataset was excluded from model selection.



Untouched test set: accuracy 0.983600, loss 0.108697 (10000 images).