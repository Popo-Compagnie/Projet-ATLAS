import pandas as pd
def test_create_dataframe():
    data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
    df = pd.DataFrame(data)
    assert df.shape == (3, 2)
    assert list(df.columns) == ['Name', 'Age']
    assert df['Age'].sum() == 90
    print("Test passed: DataFrame created successfully with correct shape and data.")
if __name__ == "__main__":
    test_create_dataframe()
# This is a test file for Git integration.
# It creates a simple DataFrame and verifies its contents.

