import torch
import matplotlib.pyplot as plt
from sklearn.metrics import \
    mean_squared_error, mean_absolute_error, r2_score, \
    explained_variance_score, mean_pinball_loss, \
    d2_pinball_score, d2_absolute_error_score


class RegressionMetrics:

    def __init__(self, y_true, y_pred, multi_output='uniform_average'):
        self.y_true = y_true
        self.y_pred = y_pred
        self.multi_output = multi_output
        self.metrics = [
            # self.abs_delta_mean,
            # self.max_delta_mean,
            # self.min_delta_mean,
            self.mean_squared_error, self.mean_absolute_error, self.r2_score,
            self.explained_variance_score, self.mean_pinball_loss,
            self.d2_pinball_score, self.d2_absolute_error_score
        ]
        self.visualize()

    def visualize(self):
        print()
        for metric in self.metrics:
            print(
                f"{metric.__name__.rjust(25, ' ')} | " +
                f"{metric()}"
            )
        # plt.plot(self.abs_delta_min(), color="green")
        # plt.plot(self.abs_delta_abs(), color="gray")
        # plt.plot(self.abs_delta_max(), color="red")
        plt.plot(self.y_true, color="green")
        plt.plot(self.y_pred, color="red")
        plt.show()

    def abs_delta_abs(self):
        return torch.abs(
            torch.abs(
                torch.FloatTensor(self.y_pred).type(torch.float32).to("cpu")
            )
            -
            torch.abs(
                torch.FloatTensor(self.y_true).type(torch.float32).to("cpu")
            )
        )

    def abs_delta_max(self):
        return torch.abs(
            torch.abs(torch.max(
                torch.FloatTensor(self.y_pred).type(torch.float32).to("cpu")
            ))
            -
            torch.abs(torch.max(
                torch.FloatTensor(self.y_true).type(torch.float32).to("cpu")
            ))
        )

    def abs_delta_min(self):
        return torch.abs(
            torch.abs(torch.min(
                torch.FloatTensor(self.y_pred).type(torch.float32).to("cpu")
            ))
            -
            torch.abs(torch.min(
                torch.FloatTensor(self.y_true).type(torch.float32).to("cpu")
            ))
        )

    def abs_delta_mean(self):
        return torch.abs(
            torch.mean(
                torch.FloatTensor(self.y_pred).type(torch.float32).to("cpu")
            )
            -
            torch.mean(
                torch.FloatTensor(self.y_true).type(torch.float32).to("cpu")
            )
        )

    def max_delta_mean(self):
        return torch.max(
            torch.mean(
                torch.FloatTensor(self.y_pred).type(torch.float32).to("cpu")
            )
            -
            torch.mean(
                torch.FloatTensor(self.y_true).type(torch.float32).to("cpu")
            )
        )

    def min_delta_mean(self):
        return torch.min(
            torch.mean(
                torch.FloatTensor(self.y_pred).type(torch.float32).to("cpu")
            )
            -
            torch.mean(
                torch.FloatTensor(self.y_true).type(torch.float32).to("cpu")
            )
        )

    def mean_squared_error(self):
        return mean_squared_error(
            self.y_true, self.y_pred, multioutput=self.multi_output
        )

    def mean_absolute_error(self):
        return mean_absolute_error(
            self.y_true, self.y_pred, multioutput=self.multi_output
        )

    def r2_score(self):
        return r2_score(
            self.y_true, self.y_pred, multioutput=self.multi_output
        )

    def explained_variance_score(self):
        return explained_variance_score(
            self.y_true, self.y_pred, multioutput=self.multi_output
        )

    def mean_pinball_loss(self):
        return mean_pinball_loss(
            self.y_true, self.y_pred, multioutput=self.multi_output
        )

    def d2_pinball_score(self):
        return d2_pinball_score(
            self.y_true, self.y_pred, multioutput=self.multi_output
        )

    def d2_absolute_error_score(self):
        return d2_absolute_error_score(
            self.y_true, self.y_pred, multioutput=self.multi_output
        )
