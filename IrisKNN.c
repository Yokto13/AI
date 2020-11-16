// Runs KNN on the Iris dataset.
// Uses a binary heap to get K-closest points.

#include <stdio.h>
#include <math.h>

double euclidean(double* p1, double* p2)
{
	double result = 0;
	for (int i = 0; i < 4; ++i)
	{
		result += pow(p1[i] - p2[i], 2);
	}
	result = pow(result, 0.5);
	return result;
}

void swap_d(int first, int second, double* collection)
{
	double temp = collection[first];
	collection[first] = collection[second];
	collection[second] = temp;
}

void swap_i(int first, int second, int* collection)
{
	int temp = collection[first];
	collection[first] = collection[second];
	collection[second] = temp;
}

int main(void)
{
	double data[150][5] =	// The IRIS DATASET in format {{Sep len, sep wid, pet len, pet wid,TARGET}}
				// The corresponding target can be found in switch at the end.
	double data_point[4];
	int k;
	printf("Please enter the followin, all in cm:\n");
	printf("Sepal length: ");
	scanf("%lf", &data_point[0]);
	printf("Sepal width: ");
	scanf("%lf", &data_point[1]);
	printf("Petal length: ");
	scanf("%lf", &data_point[2]);
	printf("Petal width: ");
	scanf("%lf", &data_point[3]);
	printf("Please enter K: ");
	scanf("%d", &k);
	// init heap
	int current_heap_size = 0;
	double heap[150];
	int index_storing[150];
	double dist;
	int node_to_check, parent;
	for (int i = 0; i < 150; ++i)
	{
		dist = euclidean(data_point, data[i]);
		heap[current_heap_size] = dist;
		index_storing[current_heap_size] = i;
		node_to_check = current_heap_size++;
		parent = node_to_check / 2;
		while (heap[node_to_check] < heap[parent])
		{
			swap_d(node_to_check, parent, heap);
			swap_i(node_to_check, parent, index_storing);
			node_to_check = parent;
			parent = node_to_check / 2;
		}
	}
	double result = 0;
	int left, right, smallest, last_smallest, input_var;
	for (int i = 0; i < k; ++i)
	{
		result += data[index_storing[0]][4];
		swap_d(0, current_heap_size - 1, heap);
		swap_i(0, current_heap_size - 1, index_storing);
		current_heap_size--;
		smallest = 1;
		do {
			left = 2 * smallest;
			right = left + 1;
			input_var = smallest;

			if (left < current_heap_size - 1 && heap[left] < heap[smallest])
				smallest = left;
			if (right < current_heap_size - 1 && heap[right] < heap[smallest])
				smallest = right;
			if (input_var != smallest)
			{
				swap_d(input_var, smallest, heap);
				swap_i(input_var, smallest, index_storing);
			}

		} while (input_var != smallest);
	}
	// averaging
	result /= k;
	printf("The prediction of %d-NN is: %.3f\n", k, result);
	printf("This prediction corresponds to: ");
	int rounded = (int)(result + 0.5);
	switch (rounded)
	{
	case 0:
		printf("Iris setosa \n");
		break;
	case 1:
		printf("Iris versicolor\n");
		break;
	case 2:
		printf("Iris virginica\n");
		break;
	default:
		printf("AN ERROR OCCURED IN THE PROGRAM \n");
		break;
	}
	return 0;
}
