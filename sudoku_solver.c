/*
 * Simple sudoku solver using backtracking.
 * 
 * Beware the solver finds solution even when multiple solution exists - it dosen't check for uniqnues. Therefore it
 * will produce an output even with empty sudoku imputed. If the sedoku can't be solved, message is shown.
 * 
 * Dominik Farhan, June 2020 
 * */

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>


void print_board(int *board)
{
	for(int i = 0; i < 81; ++i)
	{
		if(board[i] == -1)
			printf("X ");
		else
			printf("%i ", board[i]);
		if(i % 9 == 8)
			printf("\n");
	}
	printf("--------------------\n");
}

bool check_row(int* board, int index, int guess)
{
	/*
	 * Checks if a row contains the [guess].
	 * */
	int inner = index; 													/* Copy of [index] to be worked with. */
	if (inner % 9 != 8)													/* Checks row to the right. */
	{																	
		inner++;
		while (inner %9 != 0)
		{
			if (board[inner] == guess)
				return false;
			inner++;
		}
	}
	inner = index;	
	if (inner % 9 != 0)													/* Checks row to the left. */
	{																	
		inner--;
		while (inner >= 0 && inner %9 != 8)
		{
			if (board[inner] == guess)
				return false;
			inner--;
		}
	}								
	return true;
}
	

bool check_column(int* board, int index, int guess)
{
	/*
	 * Checks if a column contains [guess].
	 * */
	int inner = index - 9;
	while (inner >= 0)
	{
		if (board[inner] == guess)
			return false;
		inner -= 9;
	}
	inner = index + 9;
	while(inner < 81)
	{
		if(board[inner] == guess)
			return false;
		inner += 9;
	}
	return true;
}

int get_big_column(int index)
{
	int decider = (index % 9) / 3;										/* Gets 'off-set' (1-8), div by 3 gives col.*/
	return decider;
}

int get_big_row(int index)
{
	if(index < 27)
	{
		return 0;
	}
	else
	{
		if (index < 54)
			return 1;
		return 2;
	}
}

bool check_square(int* board, int index, int guess)
{
	/*
	 * Checks if 3x3 square in which [index] is contains guess.
	 * 
	 * The idea is to divide the board in to 3 big columns and 3 row, then any 3x3 square can be represented as indicies
	 * to those columns and rows. The indicies of values of square are then calculated from knowledge of how many places
	 * must be skipped in order to get to a different big row or big column.
	 * */
	int big_row, big_column, column_offset, row_offset;
	
	big_row = get_big_row(index);
	big_column = get_big_column(index);
	
	column_offset = big_column * 3;
	row_offset = big_row * 27;
	
	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			if(board[i + column_offset + j * 9 + row_offset] == guess)
				return false;
		}
	}
	return true;
}

bool is_possible(int *board, int index, int guess)
{
	
	/*
	 * Given [board], [index] of an empty square checks if [guess] can be placed on it.
	 */
	 
	 return	(check_row(board, index, guess) && check_column(board, index, guess) && check_square(board, index, guess));
}

int find_next_empty(int* board, int index)
{
	/*
	 * Given pointer to board and a current index it iterates trough board's elements until if finds an empty place.
	 * If it reaches the end, returns -1.
	 * 
	 * The idea of trying the next empty square is a bit dull - it dosen't use information about last filled place with
	 * much efficiency. However it is still fast enough so the program solves sudoku instantly.
	 * */
	while(index < 81 && board[index] != -1) index++;
		
	if(index == 81) 
		return -1;
	return index;
}

bool solve(int* board, int start_at)
{
	/*
	 * Recursive function to solve sudoku. Solution is placed directly to [board].
	 * 
	 * [board] is a pointer to an 9x9 array with partially unsolved sudoku.
	 * [start_at] is the first empty place in the sudoku. This place can be found with 
	 * calling [find_next_empty(board, -1)] in the first call - as is done in the main.
	 * 
	 * */
	board[start_at] = -2;
	int next = find_next_empty(board, start_at);
	for (int i = 1; i <= 9; ++i)
	{
		if(is_possible(board, start_at, i))
		{
			board[start_at] = i;
			//print_board(board);
			if(next == -1) // victory
				return true;
			if(solve(board, next))
				return true;
			board[start_at] = -2;
		}
	}
	board[start_at] = -1;
	return false;
}



int main()
{
	bool correct;
	// EASY ONE:
	int board[] = 	{-1,3,-1,-1,-1,-1,-1,-1,-1,
					-1,2,-1,9,-1,6,3,-1,-1,
					-1,6,-1,4,-1,2,-1,9,-1,
					1,-1,-1,-1,9,-1,4,-1,-1,
					-1,-1,8,1,-1,3,5,-1,-1,
					-1,-1,5,-1,7,-1,-1,-1,3,
					-1,5,-1,3,-1,1,-1,6,-1,
					-1,-1,4,6,-1,7,-1,3,-1,
					-1,-1,-1,-1,-1,-1,-1,8,-1};
	
	// HARD:
	/*
	int board[] = 	{-1,-1,2,5,-1,-1,-1,7,-1,
					-1,-1,4,1,6,-1,-1,8,-1,
					-1,5,8,-1,-1,-1,-1,4,-1,
					-1,-1,-1,-1,2,5,-1,-1,3,
					-1,-1,-1,-1,8,1,-1,-1,7,
					1,6,-1,-1,-1,-1,-1,-1,-1,
					8,-1,-1,-1,-1,7,1,-1,-1,
					-1,-1,9,3,-1,2,7,-1,-1,
					3,-1,-1,-1,-1,-1,5,-1,-1};
	*/
	
	printf("UNSOLVED:\n");
	print_board(board);
	correct = solve(board, find_next_empty(board, -1));
	if(correct)
	{
		printf("SOLUTION:\n");
		print_board(board);
	}
	else
	{
		printf("This sudoku cant be solved!\n");
	}
	
	return 0;
}
