def quicksort(arr):
    """
    Sort an array using the quicksort algorithm.
    
    Args:
        arr (list): The list to be sorted
        
    Returns:
        list: The sorted list
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

def quicksort_inplace(arr, low=0, high=None):
    """
    Sort an array in-place using the quicksort algorithm with Lomuto partition scheme.
    
    Args:
        arr (list): The list to be sorted
        low (int): Starting index
        high (int): Ending index
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition the array
        pivot_index = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)

def partition(arr, low, high):
    """
    Partition function for quicksort_inplace using Lomuto scheme.
    
    Args:
        arr (list): The array to partition
        low (int): Starting index
        high (int): Ending index
        
    Returns:
        int: The pivot index
    """
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

if __name__ == "__main__":
    # Example usage
    test_array = [3, 6, 8, 10, 1, 2, 1]
    print("Original array:", test_array)
    
    # Test the simple quicksort
    sorted_array = quicksort(test_array.copy())
    print("Sorted array (simple):", sorted_array)
    
    # Test the in-place quicksort
    inplace_array = test_array.copy()
    quicksort_inplace(inplace_array)
    print("Sorted array (in-place):", inplace_array)