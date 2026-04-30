import pandas as pd


class DateFixer:
    def __init__(self, filepath, month, year, days_in_month):
        self.filepath = filepath
        self.month = month
        self.year = year
        self.days_in_month = days_in_month
        self.df = pd.read_csv(filepath)

    def build_times(self, num_rows):
        dates = []
        hours_1 = []
        hours_2 = []

        day = 1
        row = 0
        while row < num_rows:
            date_str = f"{day:02d}/{self.month:02d}/{self.year}"
            for h in range(24):
                for q in range(4):
                    if row >= num_rows:
                        return dates, hours_1, hours_2
                    dates.append(date_str)
                    t1 = f"{h:02d}:{q*15:02d}:00"
                    next_q = (q + 1) % 4
                    next_h = h + (1 if next_q == 0 else 0)
                    t2 = f"{next_h:02d}:{next_q*15:02d}:00"
                    hours_1.append(t1)
                    hours_2.append(t2)
                    row += 1
            day += 1

        return dates, hours_1, hours_2

    def fix(self):
        all_dates = []
        all_hours_1 = []
        all_hours_2 = []

        i = 0
        while i < len(self.df):
            station = self.df.iloc[i]["counting_station"]
            direction = self.df.iloc[i]["IN_OUT"]
            count = 0
            while i + count < len(self.df) and self.df.iloc[i + count]["counting_station"] == station and self.df.iloc[i + count]["IN_OUT"] == direction:
                count += 1

            dates, hours_1, hours_2 = self.build_times(count)
            all_dates.extend(dates)
            all_hours_1.extend(hours_1)
            all_hours_2.extend(hours_2)
            i += count

        self.df["date 1"] = all_dates
        self.df["date 2"] = all_dates
        self.df["hour 1"] = all_hours_1
        self.df["hour 2"] = all_hours_2

    def save(self, output_path):
        self.df.to_csv(output_path, index=False)
        print(f"Done! Saved as {output_path}")


# February
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-02.csv",
    month=2,
    year=2024,
    days_in_month=29
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-02-fixed.csv")

# March
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-03.csv",
    month=3,
    year=2024,
    days_in_month=31
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-03-fixed.csv")

# April
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-04.csv",
    month=4,
    year=2024,
    days_in_month=30
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-04-fixed.csv")

# May
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-05.csv",
    month=5,
    year=2024,
    days_in_month=31
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-05-fixed.csv")

# June
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-06.csv",
    month=6,
    year=2024,
    days_in_month=30
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-06-fixed.csv")

# July
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-07.csv",
    month=7,
    year=2024,
    days_in_month=31
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-07-fixed.csv")

# August
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-08.csv",
    month=8,
    year=2024,
    days_in_month=31
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-08-fixed.csv")

# September
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-09.csv",
    month=9,
    year=2024,
    days_in_month=30
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-09-fixed.csv")

# October
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-10.csv",
    month=10,
    year=2024,
    days_in_month=31
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-10-fixed.csv")

# November
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-11.csv",
    month=11,
    year=2024,
    days_in_month=30
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-11-fixed.csv")

# December
fixer = DateFixer(
    filepath=r"C:\Users\anast\Downloads\data-2024-12.csv",
    month=12,
    year=2024,
    days_in_month=31
)
fixer.fix()
fixer.save(r"C:\Users\anast\Downloads\data-2024-12-fixed.csv")

