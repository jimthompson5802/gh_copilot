import dataset
import file_utils

def main():
    # Generate the synthetic regression dataset
    data = dataset.generate_synthetic_regression_dataset()

    # Save the dataset to a CSV file
    file_utils.save_dataset_to_csv(data, 'data/synthetic_regression.csv')

if __name__ == '__main__':
    main()
