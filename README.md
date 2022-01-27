# Dynamic-classifier
<P align=justify>Dynamic classifier for estimating the predictability of client's transactional behaviour
<BR>See  Alexandra Bezbochina, Elizaveta Stavinova, Anton Kovantsev and Petr Chunaev:<B> Dynamic classification of bank clients by the predictability of their transactional behavior</B>; # 166 at ICCS 2022 Main Track. </P>
<P><B>Content</B>
<LI>data_prepare.py - data preprocessing, which includes distillation of chosen categories, filling the missed weeks for each customer and calculation of indices for network training and evaluation; returns files D*table.csv and D*indtab.csv (* = 1 or 2)
<LI>dynclass_show.ipynb - notebook which performs the dynamic classificator work on each training step;
<LI>dinclass_auto - does the same without pictures, but with collecting of errors values, returns file D*increm.csv with the metrics of accuracy and AUC ROC for each step;
<LI>dinckass_noincr - shows how the system works with the incremental learning turned off and also collects of errors values, returns file D*base.csv with the metrics of accuracy and AUC ROC for each step;
<LI>compare_base - makes plots of collected errors for incremental and not incremental models;
<LI>dynclass_collect - collects the predictability data by five categories to the predictability profile, returns file D*total.csv;
<LI>profiles.ipynb - makes pictures out of collected predictability profiles.
</P>
