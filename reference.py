import material as materialwv
import json
path=materialwv.path
dataset=materialwv.dataset
total_train_img=materialwv.total_train_img
data_path=materialwv.data_path
result_path=path+'result_test\\'+dataset+'\\'


def listToSentence(lt):
    sent = ''
    for i in lt:
        sent=sent+' '+i
    return sent[:]

data_sent = json.load(open(data_path+'dataset.json'))['images']


def createReferenceValidationFile():
    for i in range(5):
        file=open(data_path+'var\\'+"ref_var"+str(i)+'.txt',"w")
        for j in range(materialwv.total_validate_img):
            file.write(listToSentence(materialwv.getSenCandidate(j+total_train_img,i)) + "\n")
        file.close()
        file=open(data_path+'var\\'+"ref_var"+str(i),"w")
        for j in range(materialwv.total_validate_img):
            file.write(listToSentence(materialwv.getSenCandidate(j+total_train_img,i)) + "\n")
        file.close()
def createReferenceTestFile():
    for i in range(5):
        file=open(result_path+"ref"+str(i)+'.txt',"w")
        for j in range(materialwv.total_test_img):
            file.write(listToSentence(materialwv.getSenCandidate(j+total_train_img+materialwv.total_validate_img,i)) + "\n")
        file.close()
        file=open(result_path+"ref"+str(i),"w")
        for j in range(materialwv.total_test_img):
            file.write(listToSentence(materialwv.getSenCandidate(j+total_train_img+materialwv.total_validate_img,i)) + "\n")
        file.close()
def main():
    #createReferenceValidationFile()
    createReferenceTestFile()
if __name__ == '__main__':
    main()
