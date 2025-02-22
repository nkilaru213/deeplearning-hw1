from typing import Tuple
import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        Initializes the WeatherForecast object by storing the dataset as a (num_days, 10) PyTorch tensor.
        Each row represents a day, and each column represents a temperature measurement.
        """
        self.data = torch.as_tensor(data_raw, dtype=torch.float32).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the minimum and maximum temperatures per day.
        Return : tuple with minimum and maximum values
        """
        return self.data.min(dim=1).values, self.data.max(dim=1).values

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest drop in average daily temperature.
        Return : the largest negative temperature difference.
        """
        daily_avg = self.data.mean(dim=1)  # calcuate the average temperature
        drop = torch.diff(daily_avg)  # calcuate the changes between consecutive days
        return drop.min()  # Figure out the most negative value 

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        Find the temperature measurement that deviates the most from the daily mean.
        Return: The most extreme temperature measurement for each day.
        """
        daily_avg = self.data.mean(dim=1, keepdim=True)  # calcuate the daily mean per day
        deviation = torch.abs(self.data - daily_avg)  # calcuate the absolute deviation
        extreme_idx = deviation.argmax(dim=1)  # Figure out the index of max deviation per day
        return self.data.gather(1, extreme_idx.unsqueeze(1)).squeeze(1)  # Retrieve extreme values


    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Calcuate the maximum temperature over the last k days.
        """
        return self.data[-k:].max(dim=1).values  # Slice last k days and get max temp

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        Predict the next day's temperature as the mean of the last k days.
        """
        return self.data[-k:].mean()  # Take mean over last k days

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        Determine the most likely day a temperature measurement was taken from dataset
        Return: Index of the day in the dataset that is closest to the given temperature.
        """
        diffs = torch.abs(self.data - t).sum(dim=1)  # Compute sum of absolute differences
        return torch.argmin(diffs)  # Return index of the closest match
