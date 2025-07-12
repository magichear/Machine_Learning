import os
import json
import bisect
from datetime import datetime


class FromYear:
    def __init__(self, data_path=None):
        """
        数据以时间戳的形式保存（方便二分）
        """
        if not data_path:
            file_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(file_path, "datasets/date_to_ids.json")
        with open(data_path, "r", encoding="utf-8") as input_file:
            self.date_to_ids = json.load(input_file)
        self.sorted_timestamps = list(map(float, self.date_to_ids))

    @staticmethod
    def __lower_bound(sorted_list, value):
        return bisect.bisect_left(sorted_list, value)

    @staticmethod
    def __upper_bound(sorted_list, value):
        return bisect.bisect_right(sorted_list, value)

    def search_timestamp(self, start_timestamp, end_timestamp):
        """
        两次二分确定范围
        """
        lower_idx = self.__lower_bound(self.sorted_timestamps, start_timestamp)
        upper_idx = self.__upper_bound(self.sorted_timestamps, end_timestamp)

        # 找范围
        range_timestamps = self.sorted_timestamps[lower_idx:upper_idx]
        if range_timestamps and range_timestamps[-1] > end_timestamp:
            range_timestamps.pop()

        # 提取ID
        results = []
        for timestamp in range_timestamps:
            results.extend(self.date_to_ids[str(timestamp)])
        return results

    def search(self, start_year, end_year):
        """
        年份包装为时间戳
        """
        start_timestamp = datetime.strptime(
            f"{start_year}-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
        ).timestamp()

        end_timestamp = datetime.strptime(
            f"{end_year}-12-31 23:59:59", "%Y-%m-%d %H:%M:%S"
        ).timestamp()

        return self.search_timestamp(start_timestamp, end_timestamp)


if __name__ == "__main__":
    from_year = FromYear()
    start_year = 2008
    end_year = 2008
    results = from_year.search(start_year, end_year)
    print(f"{start_year}年至{end_year}年的检索结果:", results)
