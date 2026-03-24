import json
import matplotlib.pyplot as plt
import os

def saveplot(xvalue,yvalues,legend,axislabel,savepath):
    """
    Plot given data
    :param xvalue: Values for x-axis
    :param yvalues: Tuple with values for y-axis
    :param legend: List with names of each yvalue for legend
    :param axislabel: List with x-axis label and y-axis label
    """
    
    plt.clf()
    pltcolors=['r','b','g','y']
    for c in range(len(yvalues)):
        
        plt.plot(xvalue,yvalues[c],pltcolors[c],label=legend[c])
    
    plt.xlabel(axislabel[0])
    plt.ylabel(axislabel[1])
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.savefig(savepath,bbox_inches = "tight")

training_results_json = ("/mnt/data/stud-uexja/DATA/vxm_twist_time_288/RESULTS/VXM_Twist_VecInt--moving+fixed--registered_NCC_500_10:43:46/VXM_Twist_VecInt--moving+fixed--registered_NCC_500_10:43:46-default-16C3-MP2-32C3-MP2-32C3-MP2-32C3-MP2-32C3-32C3-32C3-32C3-32C3-16C3-16C3-3C3/train_results_phase01.json")

with open(training_results_json) as f:
    training_dict = json.load(f)

metrics = [m for m in training_dict["training_history"] if m != "epochs"]
epochs = training_dict["training_history"]["epochs"]

sdir = ("/mnt/data/stud-uexja/Documents/MRI_Registration/CIDS_result_plot/"+
        "NCC--epoch:500--int_steps:0--bidir:False--reg_field:svf_LOW_REGULARIZAION")

if not os.path.exists(sdir):
    os.mkdir(sdir)
for metric in metrics:
    if "val_" in metric:
        continue
    else:
        metric_training_history = training_dict["training_history"][metric]
        if "val_"+metric in metrics:
            metric_validation_history = training_dict["training_history"]["val_"+metric]
            saveplot(epochs,(metric_training_history,metric_validation_history),['Training','Validation'],['epoch',metric],f"{sdir:s}/{metric:s}.png")
        else:
            saveplot(epochs,(metric_training_history),['Training'],['epoch',metric],f"{sdir:s}/{metric:s}.png")