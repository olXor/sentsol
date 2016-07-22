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
	void moveActor(WINDOW* win, float* outputs);			//length 4 array
	bool actorOnGoal();
	size_t Maze::getWallHugLength(bool hugLeft);

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

void printOutputs(WINDOW* win, float* outputs, size_t numOutputs, size_t turn, size_t leftlength, size_t rightlength, size_t mazesSolved, size_t mazesFailed, float mazeaverage);
bool positionsSame(Position p1, Position p2);

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

	size_t mazesize = 2;
	size_t mazesSolved = 0;
	size_t mazesFailed = 0;
	SentBot* bot = new SentBot(4, 4, 4, 4);
	size_t turn = 0;
	float mazeaverage = 0;
#define NUM_MAZE_AVERAGE_RESULTS 100
	size_t mazeres[NUM_MAZE_AVERAGE_RESULTS] = { 0 };
	size_t mazerespos = 0;
	float excessPain = 0.0f;
	size_t numWinsInRow = 0;
	std::ofstream avghistory("mazehistory");
	while (true) {
		Maze* maze = new Maze(mazesize, 2*mazesize);
		size_t leftlength = maze->getWallHugLength(true);
		size_t rightlength = maze->getWallHugLength(false);
		size_t largelength = std::max(leftlength, rightlength);
		maze->displayMaze(mazewin);

		size_t turnsWandering = 0;
		while (true) {
			turnsWandering++;
			turn++;
			for (size_t i = 0; i < 20; i++) {
				maze->writeActorInputs(bot->h_inputs);
				bot->takeTurn();
			}
			maze->moveActor(mazewin, bot->h_outputs);
			if (maze->actorOnGoal()) {
				float extraPleasure = 0.0;
				numWinsInRow++;
				if (excessPain > 0.0f) {
					extraPleasure = std::min(excessPain, numWinsInRow*1.0f);
				}
				excessPain -= extraPleasure + 1.0f;
				bot->givePleasurePain(1.0f + extraPleasure);
				mazesSolved++;
				if (mazesize < 30)
					mazesize++;
				break;
			}
			else if (turnsWandering >= 10 * largelength) {
				numWinsInRow = 0;
				excessPain += 1.0f;
				bot->givePleasurePain(-1.0f);
				mazesFailed++;
				if (mazesize > 2)
					mazesize--;
				break;
			}
		}
		bot->resetThoughts();
		mazeres[mazerespos] = mazesize;
		mazerespos = (mazerespos + 1) % NUM_MAZE_AVERAGE_RESULTS;
		mazeaverage = 0;
		for (size_t i = 0; i < NUM_MAZE_AVERAGE_RESULTS; i++) {
			mazeaverage += mazeres[i];
		}
		mazeaverage /= NUM_MAZE_AVERAGE_RESULTS;
		printOutputs(textwin, bot->h_outputs, 4, turnsWandering, leftlength, rightlength, mazesSolved, mazesFailed, mazeaverage);
		bot->saveWeights("maze");
		avghistory << mazeaverage << std::endl;

		delete maze;
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
	werase(win);
	int intHeight = (int)height;
	int intWidth = (int)width;
	for (int h = 0; h < intHeight; h++) {
		for (int w = 0; w < intWidth; w++) {
			Position p;
			p.x = w;
			p.y = h;
			Room* r = getRoom(p);

			mvwaddch(win, 2 * h + 2, 2 * w, '#');
			mvwaddch(win, 2 * h + 1, 2 * w + 1, ' ');
			if (r->northWall)
				mvwaddch(win, 2 * h + 2, 2 * w + 1, '#');
			else
				mvwaddch(win, 2 * h + 2, 2 * w + 1, ' ');
			if (r->westWall)
				mvwaddch(win, 2 * h + 1, 2 * w, '#');
			else
				mvwaddch(win, 2 * h + 1, 2 * w, ' ');
		}
	}
	for (int i = 0; i < 2 * intWidth; i++) {
		mvwaddch(win, 0, i, '#');
	}
	for (int i = 0; i < 2 * intHeight + 1; i++) {
		mvwaddch(win, i, 2 * intWidth, '#');
	}
	mvwaddch(win, 2 * intHeight - 1, 2 * intWidth - 1, '!');
	mvwaddch(win, 2 * actorPosition.y + 1, 2 * actorPosition.x + 1, '@');

	wrefresh(win);
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

void Maze::moveActor(WINDOW* win, float* outputs) {
	int dir = NONE;
	float maxout = -9999;
	for (size_t i = 0; i < 4; i++) {
		if (outputs[i] > maxout) {
			dir = (int)i;
			maxout = outputs[i];
		}
	}
	if (directionOpen(actorPosition, dir)) {
		mvwaddch(win, (int)(2 * actorPosition.y + 1), (int)(2 * actorPosition.x + 1), ' ');
		actorPosition = getAdjacentPosition(actorPosition, dir);
		mvwaddch(win, (int)(2 * actorPosition.y + 1), (int)(2 * actorPosition.x + 1), '@');

		wrefresh(win);
	}
}

bool Maze::actorOnGoal() {
	return positionsSame(actorPosition, goalPosition);
}

bool positionsSame(Position p1, Position p2) {
	return p1.x == p2.x && p1.y == p2.y;
}

void printOutputs(WINDOW* win, float* outputs, size_t numOutputs, size_t turn, size_t leftlength, size_t rightlength, size_t mazesSolved, size_t mazesFailed, float mazeaverage) {
	std::stringstream ss;
	ss << turn;
	mvwprintw(win, 0, 0, ss.str().c_str());

	for (size_t i = 0; i < 4; i++) {
		ss.clear();
		ss.str("");
		ss << outputs[i];

		mvwprintw(win, i + 1, 0, ss.str().c_str());
	}

	std::stringstream lss;
	lss << "L: " << leftlength;
	mvwprintw(win, 7, 0, lss.str().c_str());

	lss.clear();
	lss.str("");
	lss << "R: " << rightlength;
	mvwprintw(win, 7, 5, lss.str().c_str());

	lss.clear();
	lss.str("");
	lss << "Mazes Solved: " << mazesSolved;
	mvwprintw(win, 9, 0, lss.str().c_str());

	lss.clear();
	lss.str("");
	lss << "Mazes Failed: " << mazesFailed;
	mvwprintw(win, 10, 0, lss.str().c_str());

	lss.clear();
	lss.str("");
	lss << "Maze Level Average: " << mazeaverage;
	mvwprintw(win, 11, 0, lss.str().c_str());

	wrefresh(win);
}

int turnDir(int dir, bool left) {
	if (left) {
		if (dir == NORTH)
			return WEST;
		else if (dir == WEST)
			return SOUTH;
		else if (dir == SOUTH)
			return EAST;
		else if (dir == EAST)
			return NORTH;
	}
	else {
		if (dir == NORTH)
			return EAST;
		else if (dir == EAST)
			return SOUTH;
		else if (dir == SOUTH)
			return WEST;
		else if (dir == WEST)
			return NORTH;
	}
	return NONE;
}

size_t Maze::getWallHugLength(bool hugLeft) {
	Position p;
	p.x = 0;
	p.y = 0;
	size_t length = 0;
	int dir = NORTH;
	while (!positionsSame(p, goalPosition)) {
		dir = turnDir(dir, hugLeft);
		while (!directionOpen(p, dir)) {
			dir = turnDir(dir, !hugLeft);
		}
		p = getAdjacentPosition(p, dir);
		length++;
	}
	return length;
}