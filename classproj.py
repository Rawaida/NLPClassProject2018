from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.improve import Improved
from utils.scorer import report_score


# BASELINE Modeling - using word average length
def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev - {} test".format(language, len(data.trainset), len(data.devset), len(data.testset)))

    baseline = Baseline(language)
    
    baseline.train(data.trainset)
    
    dev = baseline.test(data.devset)
    devLabels = [sent['gold_label'] for sent in data.devset]
    
    print("Fine-tuned Score - Dev Set")
    report_score(devLabels, dev, detailed = True)
    
    predictions = baseline.test(data.testset)  
    gold_labels = [sent['gold_label'] for sent in data.testset]
    
    print("Final Score - Test Set")
    report_score(gold_labels, predictions, detailed = True)
  
    

def execute_improve (language):
    
    data = Dataset(language)
    
    print("{}: {} training - {} dev - {} test".format(language, len(data.trainset), len(data.devset), len(data.testset)))
    
    improved = Improved(language)
    
    improved.train(data.trainset)

    dev = improved.test(data.devset)
    devLabels = [sent['gold_label'] for sent in data.devset]
    
    print("Fine-tuned Score")
    report_score(devLabels, dev, detailed = True)
    
    prediction = improved.test(data.testset)  
    gold_label = [sent['gold_label'] for sent in data.testset]
    
    print("Final Score")
    report_score(gold_label, prediction, detailed = True)


if __name__ == '__main__':
    print("Baseline Model")
    print("______________")
#    execute_demo('english')
#    execute_demo('spanish')
    
    print("Improved Model")
    print("______________")
    execute_improve('english')
    execute_improve('spanish')


