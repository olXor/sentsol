#include "sentbot.cuh"
#include "curses.h"
#include <time.h>

#define NORTH 0
#define EAST 1
#define SOUTH 2
#define WEST 3
#define START 4
#define NONE -1

WINDOW* mazewin;

struct Room {
	bool northWall = true;
	bool westWall = true;
	bool visited = false;
	int backDir = NONE;
};

struct Position {
	size_t x = 99999;
	size_t y = 99999;
};

class Maze {
public:
	Maze(size_t h, size_t w);
	void displayMaze(WINDOW* win);

private:
	Room* Maze::getRoom(Position p);
	Room* Maze::getAdjacentRoom(Position p, int dir);
	Position Maze::getAdjacentPosition(Position p, int dir);
	void chiselMaze();
	void Maze::removeWall(Position p, int dir);
	std::vector<Room> maze;
	size_t height;
	size_t width;
};

int main() {
	srand((size_t)time(NULL));

	//ncurses stuff
	int err = system("mode con lines=81 cols=161");
	initscr();
	raw();
	keypad(stdscr, TRUE);
	//cbreak();
	curs_set(0);
	noecho();
	nodelay(stdscr, TRUE);

	refresh();

	mazewin = newwin(80,160,0,0);

	Maze* maze = new Maze(30,60);
	maze->displayMaze(mazewin);
	refresh();
}

Maze::Maze(size_t h, size_t w) {
	height = h;
	width = w;
	maze.resize(height*width);
	chiselMaze();
}

Room* Maze::getRoom(Position p) {
	if (p.y < height && p.x < width)
		return &maze[p.y + height*p.x];
	return NULL;
}

Room* Maze::getAdjacentRoom(Position p, int dir) {
	Position adj;
	if (dir == NORTH) {
		adj.x = p.x;
		adj.y = p.y + 1;
	}
	else if (dir == EAST) {
		adj.x = p.x + 1;
		adj.y = p.y;
	}
	else if (dir == SOUTH) {
		adj.x = p.x;
		adj.y = p.y - 1;
	}
	else if (dir == WEST) {
		adj.x = p.x - 1;
		adj.y = p.y;
	}
	else {
		throw new std::runtime_error("tried to get adjacent room with invalid direction");
	}
	return getRoom(adj);
}

void Maze::removeWall(Position p, int dir) {
	if (dir == NORTH) {
		Room* r = getRoom(p);
		if (r != NULL)
			r->northWall = false;
	}
	else if (dir == EAST) {
		Room* r = getAdjacentRoom(p, EAST);
		if (r != NULL)
			r->westWall = false;
	}
	else if (dir == SOUTH) {
		Room* r = getAdjacentRoom(p, SOUTH);
		if (r != NULL)
			r->northWall = false;
	}
	else if (dir == WEST) {
		Room* r = getRoom(p);
		if (r != NULL)
			r->westWall = false;
	}
}

Position Maze::getAdjacentPosition(Position p, int dir) {
	if (dir == NORTH)
		p.y++;
	else if (dir == EAST)
		p.x++;
	else if (dir == SOUTH)
		p.y--;
	else if (dir == WEST)
		p.x--;
	else
		throw new std::runtime_error("tried to use invalid direction to change position!");

	if (p.y < height && p.x < width)
		return p;

	throw new std::runtime_error("tried to access invalid position!");
}

int oppositeDirection(int dir) {
	if (dir == NORTH)
		return SOUTH;
	else if (dir == EAST)
		return WEST;
	else if (dir == SOUTH)
		return NORTH;
	else if (dir == WEST)
		return EAST;
	return NONE;
}

void Maze::chiselMaze() {
	Position p;
	p.x = 0;
	p.y = 0;

	Room* currentRoom = getRoom(p);
	currentRoom->backDir = START;
	currentRoom->visited = true;
	
	std::vector<int> unvisited;
	while (true) {
		unvisited.resize(0);
		for (int i = 0; i < 4; i++) {
			Room* adj = getAdjacentRoom(p, i);
			if (adj != NULL && !adj->visited)
				unvisited.push_back(i);
		}

		if (unvisited.size() > 0) {
			int nextDir = unvisited[rand() % unvisited.size()];
			removeWall(p, nextDir);
			p = getAdjacentPosition(p, nextDir);
			currentRoom = getRoom(p);
			currentRoom->visited = true;
			currentRoom->backDir = oppositeDirection(nextDir);
		}
		else if (currentRoom->backDir == START)
			break;
		else if (currentRoom->backDir == NONE) {
			throw new std::runtime_error("somehow the backdir was NONE after visiting");
		}
		else {
			p = getAdjacentPosition(p, currentRoom->backDir);
			currentRoom = getRoom(p);
		}
	}
}

void Maze::displayMaze(WINDOW* win) {
	for (size_t h = 0; h < height; h++) {
		for (size_t w = 0; w < width; w++) {
			Position p;
			p.x = w;
			p.y = h;
			Room* r = getRoom(p);

			mvaddch(2 * h + 2, 2 * w, '#');
			mvaddch(2 * h + 1, 2 * w + 1, ' ');
			if (r->northWall)
				mvaddch(2 * h + 2, 2 * w + 1, '#');
			else
				mvaddch(2 * h + 2, 2 * w + 1, ' ');
			if (r->westWall)
				mvaddch(2 * h + 1, 2 * w, '#');
			else
				mvaddch(2 * h + 1, 2 * w, ' ');
		}
	}
	for (size_t i = 0; i < 2 * width; i++) {
		mvaddch(0, i, '#');
	}
	for (size_t i = 0; i < 2 * height + 1; i++) {
		mvaddch(i, 2 * width, '#');
	}
	mvaddch(1, 1, '@');
	mvaddch(2 * height - 1, 2 * width - 1, '!');
}