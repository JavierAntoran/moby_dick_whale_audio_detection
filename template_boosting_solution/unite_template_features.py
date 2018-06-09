import numpy as np

Nspectrograms = 30000

Nfiles = 300
perfile = int(Nspectrograms/Nfiles)
Ntemplates = 300
Nfeats = 11

combine_feats = np.zeros((Nspectrograms, Ntemplates, Nfeats))

for i in range(1, Nfiles+1):

    print('working on file number %d of %d' % (i, Nfiles))
    feats = np.load('/extra/scratch03/jantoran/Documents/moby_dick/template_boosting_solution/data/features/template_features_%d.npy' % (i))
    combine_feats[(i-1)*perfile:i*perfile] = feats



combine_feats = np.reshape(combine_feats, (Nspectrograms, Ntemplates*Nfeats))
print(combine_feats.shape)

np.save('./data/template_features_combined.npy', combine_feats)