#test file to lead the pre-trained product2vec vectors at:
# http://data.dws.informatik.uni-mannheim.de/gpawdc/DL/product2vec/models/doc2vec/classification/

# specifically: classifcation-name+description-PV-DBOW-doc2vec-model-500-5-2-e500
#
# author reported full results: http://data.dws.informatik.uni-mannheim.de/gpawdc/SWJ/

from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load("/home/zz/Work/data/wop_data/classifcation-name+description-PV-DBOW-doc2vec-model-500-5-2-e500")
print("model loaded")
#print(model.docvecs['recipe__11'])
#model.docvecs.doctags.items()