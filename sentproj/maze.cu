#include "sentbot.cuh"
#include "curses.h"
#include <time.h>
#include <sstream>

#define NORTH 0
#define EAST 1
#define SOUTH 2
#define WEST 3
#define START 4
#define NONE -1

WINDOW* mazewin;
WINDOW* textwin;

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
	void writeActorInputs(float* inputs);	//length 4 array
	void moveActor(float* outputs);			//length 4 array
	bool actorOnGoal();

private:
	Room* getRoom(Position p);
	Room* getAdjacentRoom(Position p, int dir);
	Position getAdjacentPosition(Position p, int dir);
	void chiselMaze();
	void removeWall(Position p, int dir);
	bool directionOpen(Position p, int dir);
	std::vector<Room> maze;
	size_t height;
	size_t width;

	Position actorPosition;
	Position goalPosition;
};

void printOutputs(WINDOW* win, float* outputs, size_t numOutputs, size_t turn);

int main() {
	srand((unsigned int)time(NULL));

	//ncurses stuff
	int err = system("mode con lines=81 cols=201");
	initscr();
	raw();
	keypad(stdscr, TRUE);
	//cbreak();
	curs_set(0);
	noecho();
	nodelay(stdscr, TRUE);

	refresh();

	mazewin = newwin(80, 160, 0, 0);
	textwin = newwin(80, 40, 0, 160);

	SentBot* bot = new SentBot(4, 4, 4, 4);
	while (true) {
		Maze* maze = new Maze(30, 60);
		maze->displayMaze(mazewin);
		refresh();

		size_t turnsWandering = 0;
		while (true) {
			turnsWandering++;
			if (turnsWandering % 1000 == 0) {
				printOutputs(textwin, bot->h_outputs, 4, turnsWandering);
				bot->saveWeights("maze");
			}
			maze->writeActorInputs(bot->h_inputs);
			bot->takeTurn();
			maze->moveActor(bot->h_outputs);
			if (maze->actorOnGoal()) {
				bot->givePleasurePain(1);
				break;
			}
			else if (turnsWandering % 10000 == 0) {
				bot->givePleasurePain(-1);
			}
		}
	}
}

Maze::Maze(size_t h, size_t w) {
	height = h;
	width = w;
	maze.resize(height*width);
	chiselMaze();
	goalPosition.x = w - 1;
	goalPosition.y = h - 1;
	actorPosition.x = 0;
	actorPosition.y = 0;
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

bool Maze::directionOpen(Position p, int dir) {
	if (dir == NORTH) {
		Room* r = getRoom(p);
		return r != NULL && !r->northWall;
	}
	else if (dir == EAST) {
		Room* r = getAdjacentRoom(p, EAST);
		return r != NULL && !r->westWall;
	}
	else if (dir == SOUTH) {
		Room* r = getAdjacentRoom(p, SOUTH);
		return r != NULL && !r->northWall;
	}
	else if (dir == WEST) {
		Room* r = getRoom(p);
		return r != NULL && !r->westWall;
	}
	return false;
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
	int intHeight = (int)height;
	int intWidth = (int)width;
	for (int h = 0; h < intHeight; h++) {
		for (int w = 0; w < intWidth; w++) {
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
	for (int i = 0; i < 2 * intWidth; i++) {
		mvaddch(0, i, '#');
	}
	for (int i = 0; i < 2 * intHeight + 1; i++) {
		mvaddch(i, 2 * intWidth, '#');
	}
	mvaddch(2 * intHeight - 1, 2 * intWidth - 1, '!');
	mvaddch(2 * actorPosition.y + 1, 2 * actorPosition.x + 1, '@');
}

void Maze::writeActorInputs(float* inputs) {
	bool open = directionOpen(actorPosition, NORTH);
	if (open)
		inputs[0] = 1.0f;
	else
		inputs[0] = -1.0f;
	open = directionOpen(actorPosition, EAST);
	if (open)
		inputs[1] = 1.0f;
	else
		inputs[1] = -1.0f;
	open = directionOpen(actorPosition, SOUTH);
	if (open)
		inputs[2] = 1.0f;
	else
		inputs[2] = -1.0f;
	open = directionOpen(actorPosition, WEST);
	if (open)
		inputs[3] = 1.0f;
	else
		inputs[3] = -1.0f;
}

void Maze::moveActor(float* outputs) {
	int dir = NONE;
	float maxout = -9999;
	for (size_t i = 0; i < 4; i++) {
		if (outputs[i] > maxout) {
			dir = (int)i;
			maxout = outputs[i];
		}
	}
	if (maxout > 1.0f && directionOpen(actorPosition, dir)) {
		mvaddch((int)(2 * actorPosition.y + 1), (int)(2 * actorPosition.x + 1), ' ');
		actorPosition = getAdjacentPosition(actorPosition, dir);
		mvaddch((int)(2 * actorPosition.y + 1), (int)(2 * actorPosition.x + 1), '@');
	}
}

bool Maze::actorOnGoal() {
	return actorPosition.x == goalPosition.x && actorPosition.y == goalPosition.y;
}

void printOutputs(WINDOW* win, float* outputs, size_t numOutputs, size_t turn) {
	int startx;
	int starty;
	getbegyx(win, starty, startx);
	std::stringstream ss;
	ss << turn;
	mvprintw(starty, startx, ss.str().c_str());

	for (size_t i = 0; i < 4; i++) {
		ss.clear();
		ss.str("");
		ss << outputs[i];

		mvprintw(starty + i + 1, startx, ss.str().c_str());
	}
	refresh();
}
