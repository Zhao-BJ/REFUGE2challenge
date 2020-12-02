import pandas as pd


v1_dir = 'H:/Glaucoma/REFUGE/submit_results/review/v1/classification_results.csv'
v2_dir = 'H:/Glaucoma/REFUGE/submit_results/review/v2/classification_results.csv'
v3_dir = 'H:/Glaucoma/REFUGE/submit_results/review/v3/classification_results.csv'
v4_dir = 'H:/Glaucoma/REFUGE/submit_results/review/v4/classification_results.csv'
v5_dir = 'H:/Glaucoma/REFUGE/submit_results/review/v5/classification_results.csv'
new_dir = 'H:/Glaucoma/REFUGE/submit_results/review/classification_results.csv'

p1 = pd.read_csv(v1_dir)
p2 = pd.read_csv(v2_dir)
p3 = pd.read_csv(v3_dir)
p4 = pd.read_csv(v4_dir)
p5 = pd.read_csv(v5_dir)
print(p5.shape)


new_score = []
for i in range(p5.shape[0]):
    print(p5['FileName'].iloc[i])
    va1 = p1['Glaucoma Risk'].iloc[i]
    va2 = p2['Glaucoma Risk'].iloc[i]
    va3 = p3['Glaucoma Risk'].iloc[i]
    va4 = p4['Glaucoma Risk'].iloc[i]
    va5 = p5['Glaucoma Risk'].iloc[i]
    
    new_va = (va1 + va2 + va3 + va4 + va5) / 5
    new_score.append(new_va)
pnew = pd.DataFrame({'FileName': p5['FileName'].values, 'Glaucoma Risk': new_score})
pnew.to_csv(new_dir)