# Dynamic-classifier
<P align=justify>Dynamic classifier for estimating the predictability of client's transactional behaviour
<BR>See  Alexandra Bezbochina, Elizaveta Stavinova, Anton Kovantsev and Petr Chunaev:<B> Dynamic classification of bank clients by the predictability of their transactional behavior</B>; # 166 at ICCS 2022 Main Track. </P>
<P><B>Content</B>
<LI>data_prepare.py - data preprocessing, which includes distillation of chosen categories, filling the missed weeks for each customer and calculation of indices for network training and evaluation;
<LI>dynclass_show.ipynb - notebook which performs the dynamic classificator work on each training step;
<LI>dinclass_auto - does the same without pictures, but with collecting of errors values for each step;
<LI>dinckass_noincr - shows how the system works with the incremental learning turned off and also collects of errors values for each step;
<LI>compare_base - makes plots of collected errors for incremental and not incremental models;
</P>
