# Extracts the features, labels, and normalizes the training and test split features. Make sure you update the location
# of the downloaded datasets before in the cls_feature_class.py

import cls_feature_class

dataset_name = 'ansim'  # Datasets: ansim, resim, cansim, cresim, real, mansim and mreal

# Extracts feature and labels for all overlap and splits
for ovo in [1, 2, 3]:  # SE overlap. Change to [1] if you are only calculating the features for overlap 1.
    for splito in [1, 2, 3]:    # all splits. Use [1, 8, 9] for 'real' and 'mreal' datasets. Change to [1] if you are only calculating features for split 1.
        for nffto in [512]: # For now use 512 point FFT. Once you get the code running, you can play around with this.
            feat_cls = cls_feature_class.FeatureClass(ov=ovo, split=splito, nfft=nffto, dataset=dataset_name)

            # Extract features and normalize them
            feat_cls.extract_all_feature()
            feat_cls.preprocess_features()

            # # Extract labels in regression mode
            feat_cls.extract_all_labels('regr', 0)
