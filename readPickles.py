#import numpy
import pprint, cPickle, csv

ds=[]
a=0
# read system predictions
with open('predictions.csv', 'rb') as csvfile:
   spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
   for row in spamreader:
      #print row	
      ds.append(row)

#pprint.pprint(ds)

# read template file 
pkl_file = open('prediction_template_test.pkl', 'rb')

data1 = cPickle.load(pkl_file)
d=0;
for key1 in data1:
   d=d+1
   i=0
   for key2 in data1[key1]:
      i=i+1
      data1[ds[0][d]][ds[i][0]]=float(ds[i][d])

pprint.pprint(data1)

pkl_file.close()
# write pkl file
f = file('predictions.pkl', 'wb')
cPickle.dump(data1, f, protocol=cPickle.HIGHEST_PROTOCOL)
#cPickle.dump(data1, f, protocol=2)
f.close()


