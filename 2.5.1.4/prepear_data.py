import datetime
import csv
import subprocess


def speedtest():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    speedtest_cmd = "speedtest-cli --simple"
    process = subprocess.Popen(speedtest_cmd.split(), stdout=subprocess.PIPE)
    process_output = process.communicate()[0]
    process_output = process_output.decode("utf-8")
    process_output = process_output.split()
    process_output.append(date_time)
    return process_output


def save_to_csv(data, filename):
    try:
        with open(filename + ".csv", "a") as f:
            wr = csv.writer(f)
            wr.writerow(data)
    except:
        with open(filename + ".csv", "w") as f:
            wr = csv.writer(f)
            wr.writerow(data)


def print_from_csv(filename):
    with open(filename + ".csv", "r") as f:
        re = csv.reader(f)
        for line in re:
            print(line)


if __name__ == "__main__":

    #    for i in range(5):
    #        speedtest_output = speedtest()
    #        print('Test number {}'.format(i))
    #        print(speedtest_output)
    #        save_to_csv(speedtest_output, '/tmp/rpi_data_test')
    print_from_csv("/tmp/rpi_data_test")
