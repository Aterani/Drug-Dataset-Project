tp_m = int(input("input male tp"))
fp_m = int(input("input male fp"))
tn_m = int(input("input male tn"))
fn_m = int(input("input male fn"))

tp_f = int(input("input female tp"))
fp_f = int(input("input female fp"))
tn_f = int(input("input female tn"))
fn_f = int(input("input female fn"))

ppr_m = (tp_m + fp_m) / (tp_m + fp_m + tn_m + fn_m)
ppr_f = (tp_f + fp_f) / (tp_f + fp_f + tn_f + fn_f)


tpr_m = tp_m/(tp_m + fn_m)
tpr_f = tp_f/(tp_f + fn_f)

fpr_m = fp_m/(fp_m+tn_m)
fpr_f = fp_f/(fp_f+tn_f)
print("male tpr = " + str(tpr_m))
print("female tpr = " + str(tpr_f))
print("dp = " + str(abs(ppr_m-ppr_f)))
print("EOp = " + str(abs(tpr_m-tpr_f)))
print("EOd = " + str(1/2 * (abs(tpr_m-tpr_f) + abs(fpr_m-fpr_f))))
print("accuracy = " + str((tp_m + tn_m + tp_f + tn_f)/(tp_m + tn_m + tp_f + tn_f + fp_m + fn_m + fp_f + fn_f)))