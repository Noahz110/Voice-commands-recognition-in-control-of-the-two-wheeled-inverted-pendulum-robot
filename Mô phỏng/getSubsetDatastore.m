function dsSubset = getSubsetDatastore(ads,indices)

dsSubset = copy(ads);
dsSubset.Files  = ads.Files(indices);
dsSubset.Labels = ads.Labels(indices);

end