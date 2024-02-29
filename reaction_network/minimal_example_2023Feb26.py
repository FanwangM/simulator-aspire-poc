#
# %%
import time
from collections import OrderedDict
import itertools as it

import pandas as pd

from fjss import FJS2
from utils import *  # get_m_value, parse_data
from profiling_utils import *

# %%
input_fname = "gfjsp_10_5_1.txt"
infinity = 1.0e7
n_opt_selected = 30
num_workers = 0
verbose = False
# %%

new_row = OrderedDict()
new_row["method"] = "MILP"

print("loading and setting up data")
(
    n_opt,
    n_mach,
    operations,
    machines,
    operation_data,
    machine_data,
    para_lmin,
    para_lmax,
    para_p,
    para_h,
    para_w,
    para_delta,
    para_a,
    para_mach_capacity,
) = prepare_input(method="milp", n_opt_selected=n_opt_selected, input_fname=input_fname)
para_a = check_fix_shape_of_para_a(para_p, para_a, intended_for="milp")
new_row["n_opt"] = n_opt
new_row["n_mach"] = n_mach

# %%
print("solve the MILP problem with FJS2")

# check the running time
start_time = time.time()
fjss2 = FJS2(
    operations=operations,
    machines=machines,
    para_p=para_p,
    para_a=para_a,
    para_w=para_w,
    para_h=para_h,
    para_delta=para_delta,
    para_mach_capacity=para_mach_capacity,
    # para_lmin=para_lmin_new,
    para_lmin=para_lmin,
    # para_lmax=np.full_like(para_lmax, np.inf),
    para_lmax=para_lmax,
    precedence=None,
    model_string=None,
    inf_milp=infinity,
    workshifts=None,
    # workshifts=[(500, 0)] * 10, # violates the lag time constraint
    # workshifts=[(400, 5)] * 10, # this works because the lag time constraint is not violated
    # workshifts=[(374, 5)] * 10,
    # workshifts=[(400, 0)] * 10,

    # num_workshifts=None, # not used
    # shift_durations=500, # works
    # shift_durations=70, # infeasible

    # shift_durations=400, # works
    # shift_durations=1000, # sees the effect of the constraint
    # shift_durations=374, # 324 is the limit
    # shift_durations=375,
    # shift_durations=323,

    # shift_durations=270, # infeasible
    # shift_durations=2000, # works
    # shift_durations=1650, # works
    operations_subset_indices=None,
    num_workers=num_workers,
    verbose=verbose,
    big_m=None,
)
fjss2.build_model_gurobi()
fjss2_output = fjss2.solve_gurobi()
end_time = time.time()
running_time_seconds = end_time - start_time
new_row["running_time_seconds"] = running_time_seconds
print("checking if the solution satisfies the constraints of MILP")
model = fjss2.model
# get the number of constraints
new_row["num_constraints"] = model.NumConstrs
# get the number of variables
new_row["num_variables"] = model.NumVars
# makespan
makespan = model.objVal
new_row["makespan"] = makespan
var_x = fjss2.var_x.X
var_y = fjss2.var_y.X
var_z = fjss2.var_z.X
var_s = fjss2.var_s.X
var_c = fjss2.var_c.X
var_c_max = fjss2.var_c_max.X
big_m = fjss2.big_m
try:
    para_a = check_fix_shape_of_para_a(fjss2.para_p, fjss2.para_a, intended_for="milp")
    check_constraints_milp(
        var_y=var_y,
        var_s=var_s,
        var_c=var_c,
        var_c_max=var_c_max,
        operations=operations,
        machines=machines,
        para_p=fjss2.para_p,
        para_a=para_a,
        para_w=fjss2.para_w,
        para_h=fjss2.para_h,
        para_delta=fjss2.para_delta,
        para_mach_capacity=fjss2.para_mach_capacity,
        para_lmin=fjss2.para_lmin,
        para_lmax=fjss2.para_lmax,
        big_m=big_m,
        var_x=var_x,
        var_z=var_z,
    )
    print("the solution satisfies the constraints of MILP formulation.")
    new_row["feasible_MILP"] = "yes"
except:  # pylint: disable=bare-except
    print("the solution does not satisfy the constraints of MILP formulation.")
    new_row["feasible_MILP"] = "no"


print(f"data record: {new_row}")


from fjss import FjsOutput, SolvedOperation

assignments = dict()
start_times = dict()
end_times = dict()
solved_operations = []

# based on https://github.com/FanwangM/solver_checking/blob/6d596f4741482e511f29680b991a0d5c0228391a/debugging_CP.ipynb#L2133
df = pd.DataFrame(
    columns=["Task", "Resource", "Start", "Finish"])

operations_ids = list(fjss2.operations.keys())
machine_ids = list(fjss2.machines.keys())

for i, m in it.product(range(len(fjss2.operations)), range(len(fjss2.machines))):
    if var_y[i, m] == 1:
        # TODO: fix this because operations is not a taks
        # a task is a set of operations
        df.loc[i, "Task"] = operations_ids[i]
        # df.loc[i, "Task"] =
        df.loc[i, "Resource"] = machine_ids[m]
        df.loc[i, "Start"] = var_s[i]
        df.loc[i, "Finish"] = var_c[i]

# %%
import plotly.express as px
from plotly.figure_factory import create_gantt

# sort the dataframe by "Resource" column which is treated as integer
df["Task"] = df["Task"].astype(int)
df = df.sort_values(by="Task", ascending=False)

# get the colors from plotly, Dark24
color_list = px.colors.qualitative.Dark24[:6]

colors_dict = {
    0: color_list[0],
    1: color_list[1],
    2: color_list[2],
    3: color_list[3],
    4: color_list[4],
    5: color_list[5],
}

fig = create_gantt(
    df,
    # colors='Blues',
    index_col="Resource",
    show_colorbar=True,
    bar_width=0.5,
    showgrid_x=True,
    showgrid_y=True,
    group_tasks=True,
    colors=colors_dict,
)
fig.update_layout(xaxis_type="linear", autosize=True, width=800, height=600)

# add y axis label with "operations"
fig.update_layout(yaxis_title="Operations")
# add x axis label with "time"
fig.update_layout(xaxis_title="Time")

# # save the figure
# fig.write_image("gantt_chart_MILP_30opt_400_5_ws_2024Feb25_v1.png")

fig.show()

## the missing of machine #4 in the case of 400_0 was caused by the fact that machine 3 and 4 can do the same set of operations, that's
# (array([ 1,  3,  5,  7, 10, 12, 14, 16, 18, 21, 23, 25, 27]),)
