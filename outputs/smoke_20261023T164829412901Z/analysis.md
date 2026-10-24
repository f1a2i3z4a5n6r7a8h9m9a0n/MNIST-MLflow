# SMOKE TEST ONLY



Tracking mode: local. Local runs are not DagsHub evidence.



Selected final-epoch model: **experiment2_256_adam_b64**, using validation accuracy first and lower validation loss to break ties.

Validation accuracy: 0.738281; validation loss: 1.023063.

The next-ranked model, experiment3_128_64_adam_b64, has validation accuracy 0.699219 and validation loss 1.425894. These observed validation values determine the ranking, rather than model size or training accuracy.

Training minus validation accuracy: -0.211914. Minimum validation loss occurred at epoch 1.



Final validation loss is 0.000000 above its minimum. Since that epoch, training loss changed by 0.000000. This particular loss comparison does not show simultaneous increasing validation loss and decreasing training loss; inspect all curves and the accuracy gap as well.

A training/validation gap, especially when training loss falls while validation loss rises, is evidence of possible overfitting. The curves describe this sample; one seed does not establish that an architecture is always superior. Training metrics are the last fit epoch's batch aggregates; validation metrics evaluate the final epoch weights.



All models use final epoch weights; there is no early stopping or best-epoch restoration. The test dataset was excluded from model selection.



Smoke runs use 1,024 training images, 256 validation images and one epoch. Test evaluation is skipped; these are not assignment results.