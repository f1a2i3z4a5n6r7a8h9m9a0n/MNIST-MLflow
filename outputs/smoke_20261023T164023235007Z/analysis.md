# SMOKE TEST ONLY



Tracking mode: local. Local runs are not DagsHub evidence.



Selected final-epoch model: **experiment2_256_adam_b64**, using validation accuracy first and lower validation loss to break ties.

Validation accuracy: 0.738281; validation loss: 1.023063.

Training minus validation accuracy: -0.211914. Minimum validation loss occurred at epoch 1.



A training/validation gap, especially when training loss falls while validation loss rises, is evidence of possible overfitting. The curves describe this sample; one seed does not establish that an architecture is always superior. Training metrics are the last fit epoch's batch aggregates; validation metrics evaluate the final epoch weights.



All models use final epoch weights; there is no early stopping or best-epoch restoration. The test dataset was excluded from model selection.



Smoke runs use 1,024 training images, 256 validation images and one epoch. Test evaluation is skipped; these are not assignment results.