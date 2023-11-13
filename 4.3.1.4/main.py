import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


filename = "./Data/internet_traffic_hist.csv"
df_hist = pd.read_csv(filename)
print(df_hist.head)

plt.figure(figsize=(20, 10))

# XY Plot of year and traffic
x = df_hist.year
y = df_hist.traffic

# XY Plot of year and traffic
plt.plot(x, y, label=" ", linewidth=7)
plt.plot(x, y, "*k", markersize=14, label="")

# Increase sligthly the axis sizes to make the plot more clear
plt.axis([x.iloc[0] - 1, x.iloc[-1] + 1, y.iloc[0] * -0.1, y.iloc[-1] * 1.1])

# Add axis labels
plt.xlabel("Year")
plt.ylabel("Fixed Internet Traffic Volume")

# Increase default font size
plt.rcParams.update({"font.size": 26})
plt.show()


plt.figure(figsize=(20, 10))

order = 1

# XY Plot of year and traffic
x = df_hist.year
y = df_hist.traffic

m, b = np.polyfit(x, y, order)

plt.plot(x, y, label="Historical Internet Traffic", linewidth=7)
plt.plot(x, y, "*k", markersize=15, label="")
plt.plot(x, m * x + b, "-", label="Simple Linear Regression Line", linewidth=6)

print("The slope of line is {}.".format(m))
print("The y intercept is {}.".format(b))
print("The best fit simple linear regression line is {}x + {}.".format(m, b))


# Increase sligthly the axis sizes to make the plot more clear
plt.axis([x.iloc[0] - 1, x.iloc[-1] + 1, y.iloc[0] * -0.1, y.iloc[-1] * 1.1])

# Add axis labels
plt.xlabel("Year")
plt.ylabel("Fixed Internet Traffic Volume")
plt.legend(loc="upper left")

# Increase default font size
plt.rcParams.update({"font.size": 26})
plt.show()

models = []  # to store polynomial model parameters (list of poly1d objects)
errors_hist = (
    []
)  # to store the absolute errors for each point (2005-2015) and for each model (list of numpy arrays )
mse_hist = []  # to store the MSE for each model (list of numpy floats)

# Try polynomial models with increasing order
for order in range(1, 4):
    # Fit polynomial model
    p = np.poly1d(np.polyfit(x, y, order))
    models.append(p)

plt.figure(figsize=(20, 10))

# Visualize polynomial models fit
for model in models[0:3]:
    plt.plot(x, model(x), label="order " + str(len(model)), linewidth=7)

plt.plot(x, y, "*k", markersize=14, label="Historical Internet Traffic", linewidth=7)
plt.legend(loc="upper left")

# Add axis labels
plt.xlabel("Year")
plt.ylabel("Fixed Internet Traffic Volume")

plt.show()


models = []  # to store polynomial model parameters (list of poly1d objects)
errors_hist = (
    []
)  # to store the absolute errors for each point (2005-2015) and for each model (list of numpy arrays )
mse_hist = []  # to store the MSE for each model (list of numpy floats)

# Try polynomial models with increasing order
for order in range(1, 4):
    # Fit polynomial model
    p = np.poly1d(np.polyfit(x, y, order))
    models.append(p)

    e = np.abs(y - p(x))  # absolute error
    mse = np.sum(e**2) / len(df_hist)  # mse

    errors_hist.append(e)  # Store the absolute errors
    mse_hist.append(mse)  # Store the mse

plt.show()


# Visualize fit error for each year
x = df_hist.year
width = 0.2  # size of the bar

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

p1 = ax.bar(x, errors_hist[0], width, color="b", label="Abs. error order 1 fit")
p2 = ax.bar(x + width, errors_hist[1], width, color="r", label="Abs. error order 2 fit")
p3 = ax.bar(
    x + 2 * width, errors_hist[2], width, color="y", label="Abs. error order 3 fit"
)

# "Prettyfy" the bar graph
ax.set_xticks(x + 2 * width)
ax.set_xticklabels(x)
plt.legend(loc="upper left", fontsize=16)
plt.show()

# Visualise MSE for each model
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

x = np.array([0, 1, 2])
width = 0.6  # size of the bar

p1 = ax.bar(x[0], mse_hist[0], width, color="b", label="pred. error order 1 fit")
p2 = ax.bar(x[1], mse_hist[1], width, color="r", label="pred. error order 2 fit")
p3 = ax.bar(x[2], mse_hist[2], width, color="y", label="pred. error order 3 fit")

ax.set_xticks(x + width / 2)
ax.set_xticklabels(["Poly. order 1", "Poly. order 2", "Poly. order 3"], rotation=90)
plt.show()

# Polynomial function order
order = 3

x = df_hist.year.values  # regressor
y = df_hist.traffic.values  # regressand

# Fit the model, return the polynomial parameter values in a numpy array such that
# y = p[0]*x**order + p[1]*x*(order-1) ...

p_array = np.polyfit(x, y, order)

print(type(p_array), p_array)

# poly1d is a convenience class, used to encapsulate “natural” operations on polynomials
# so that said operations may take on their customary form in code

# wrap the p_array in a poly1 object
p = np.poly1d(p_array)

print(type(p), p)

# use the poly1d object to evaluate the value of the polynomial in a specific point
print("The value of the polynomial for x = 2020 is : {} ".format(p(2020)))

# compute the absolute error for each value of x and the MSE error for the estimated polynomial model
e = np.abs(y - p(x))
mse = np.sum(e**2) / len(x)

print("The estimated polynomial parameters are: {}".format(p))
print(
    "The errors for each value of x, given the estimated polynomial parameters are: \n {}".format(
        e
    )
)
print("The MSE is :{}".format(mse))


# Non linear regression model fitting

# First, define the regression model function, in this case, we'll choose an exponential of the form y= a*(b^(x))
def my_exp_func(x, a, b):
    return a * (b**x)


x = np.arange(
    2016 - 2005
)  # the regressor is not the year in itself, but the number of years from 2005
y = df_hist.traffic.values  # regressand

# use curve_fit to find the exponential parameters vector p. cov expresses the confidence of the
# algorithm on the estimated parameters
p, cov = curve_fit(my_exp_func, x, y)
e = np.abs(y - my_exp_func(x, *p))
mse = np.sum(e**2) / len(df_hist)

print("The estimated exponential parameters are: {}".format(p))
print(
    "The errors for each value of x, given the estimated exponential parameters are: \n {}".format(
        e
    )
)
print("The MSE is :{}".format(mse))

models.append(p)

errors_hist.append(e)  # Store the absolute error
mse_hist.append(mse)

# Compare the errors and visualize the fit for the different regression models.
plt.figure(figsize=(20, 10))

# Visualize polynomial models fit
for model in models[0:-1]:
    x = df_hist.year.values
    y = df_hist.traffic.values
    plt.plot(x, model(x), label="order " + str(len(model)), linewidth=7)

# Visualize exponenetial model fit
x = np.arange(
    2016 - 2005
)  # the regressor is not the year in itself, but the number of years from 2005
plt.plot(
    df_hist.year.values,
    my_exp_func(x, *models[-1]),
    label="Exp. non-linear regression",
    linewidth=7,
)

plt.plot(
    df_hist.year,
    df_hist.traffic,
    "*k",
    markersize=14,
    label="Historical Internet Traffic",
)
plt.legend(loc="upper left")

# Add axis labels
plt.xlabel("Year")
plt.ylabel("Fixed Internet Traffic Volume")
plt.show()


# Visualize fit error for each year

x = df_hist.year
width = 0.2  # size of the bar

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

p1 = ax.bar(x, errors_hist[0], width, color="b", label="Abs. error order 1 fit")
p2 = ax.bar(x + width, errors_hist[1], width, color="r", label="Abs. error order 2 fit")
p3 = ax.bar(
    x + 2 * width, errors_hist[2], width, color="y", label="Abs. error order 3 fit"
)
p4 = ax.bar(
    x + 3 * width, errors_hist[3], width, color="g", label="Abs. exponential fit"
)

# "Prettyfy" the bar graph
ax.set_xticks(x + 2 * width)
ax.set_xticklabels(x)
plt.legend(loc="upper left", fontsize=16)
plt.show()

# Visualise MSE for each model
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

x = np.array([0, 1, 2, 3])
width = 0.6  # size of the bar

p1 = ax.bar(x[0], mse_hist[0], width, color="b", label="pred. error order 1 fit")
p2 = ax.bar(x[1], mse_hist[1], width, color="r", label="pred. error order 2 fit")
p3 = ax.bar(x[2], mse_hist[2], width, color="y", label="pred. error order 3 fit")
p4 = ax.bar(x[3], mse_hist[3], width, color="g", label="pred. exponential fit")

ax.set_xticks(x + width / 2)
ax.set_xticklabels(
    ["Poly. order 1", "Poly. order 2", "Poly. order 3", "Exp. model"], rotation=90
)
plt.show()

filename = "./Data/internet_traffic_proj.csv"
df_proj = pd.read_csv(filename)
print(df_proj.head())

# Compare linear and nonlinear model prediction errors.
df = pd.concat([df_hist, df_proj]).reset_index()
df.drop("index", axis=1, inplace=True)
df = df.drop_duplicates()  # The 2015 value is found in both the df_hist and df_proj df
df.head(20)

plt.figure(figsize=(20, 10))

errors_all = []
mse_all = []

for model in models[0:-1]:
    # Visualize polynomial model fit
    x = df.year.values
    y = df.traffic.values
    plt.plot(x, model(x), label="order " + str(len(model)), linewidth=7)

    # error and mse for polynomial models
    pred_y = model(x)
    e = np.abs(y - pred_y)
    errors_all.append(e)  # Store the absolute errors
    mse_all.append(np.sum(e**2) / len(df))  # Store the mse

# Visualize exponential model fit
x = np.arange(
    2021 - 2005
)  # the regressor is not the year in itself, but the number of years from 2005
pred_y = my_exp_func(x, *models[-1])
plt.plot(df.year.values, pred_y, label="Exp. non-linear regression", linewidth=7)

# errors and mse for exponential model
e = np.abs(y - pred_y)
errors_all.append(e)  # Store the absolute errors
mse_all.append(np.sum(e**2) / len(df))  # Store the mse

plt.plot(df.year, df.traffic, "*k", markersize=14, label="Projected Internet Traffic")
plt.legend(loc="upper left")

# Add axis labels
plt.xlabel("Year")
plt.ylabel("Fixed Internet Traffic Volume")
plt.axis([2004, 2021, -300, 3500])
plt.show()
