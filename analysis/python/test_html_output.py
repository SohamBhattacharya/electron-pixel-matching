#import plotly.express as px
import matplotlib.pyplot as plt, mpld3

mpl_fig, mpl_axis = plt.subplots()

mpl_axis.scatter(x=range(10), y=range(10), c = "r")
mpl_axis.scatter(x=range(10, 20), y=range(10, 20), c = "b")

mpld3.save_html(mpl_fig, "plots/test_plotly.html")

print("Done")