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

#define NUM_BOT_TURNS_PER_MOVE 100

#define USE_EXTRA_PLEASURE

//#define OPPOSING_OUTPUTS	//2 outputs instead of 4; left-right and up-down are opposing each other on the same bit

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
	bool moveActor(WINDOW* win, float* outputs, size_t* moveTotals);			//length 4 array
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

void printOutputs(WINDOW* win, float* outputs, size_t numOutputs, size_t moves, size_t turn, size_t level, size_t leftlength, size_t rightlength, size_t mazesSolved, size_t mazesFailed, float mazeaverage, size_t* moveTotals);
bool positionsSame(Position p1, Position p2);

float pleasurePainLengthMultiplier(size_t length) {
	return 1.0f/(1.0f - (float)pow(VALUE_DECAY_FACTOR, NUM_BOT_TURNS_PER_MOVE*length));
}

int main() {
	srand((unsigned int)time(NULL));

	//ncurses stuff
	int err = system("mode con lines=81 cols=201");
	initscr();
	raw();
	keypad(stdscr, TRUE);
	cbreak();
	curs_set(0);
	noecho();
	nodelay(stdscr, TRUE);
	timeout(-1);

	refresh();

	mazewin = newwin(80, 160, 0, 0);
	textwin = newwin(80, 40, 0, 160);

	size_t mazesize = 2;
	size_t mazesSolved = 0;
	size_t mazesFailed = 0;

#ifdef OPPOSING_OUTPUTS
	size_t numOutputs = 2;
#else
	size_t numOutputs = 4;
#endif
	SentBot* bot = new SentBot(4, numOutputs, 4, 4);

	mvwprintw(mazewin, 0, 0, "Do you want to start a new weights file (overwriting the old one)?");
	wrefresh(mazewin);

	std::ofstream avghistory;

	int ch = getch();
	if (!(ch == 'y' || ch == 'Y')) {
		bot->loadWeights("maze");
		avghistory.open("mazehistory", std::ios::app);
	}
	else {
		avghistory.open("mazehistory");
	}

	size_t turn = 0;
	float mazeaverage = 0;
#define NUM_MAZE_AVERAGE_RESULTS 100
	size_t mazeres[NUM_MAZE_AVERAGE_RESULTS] = { 0 };
	size_t mazerespos = 0;
	float excessPain = 0.0f;
	size_t numWinsInRow = 0;
	size_t turnsWandering = 0;
	size_t moves = 0;
	size_t moveTotals[4] = { 0 };
	float* convertedOutputs;
#ifndef OPPOSING_OUTPUTS
	convertedOutputs = bot->h_outputs;
#else
	convertedOutputs = new float[4];
#endif

	while (true) {
		Maze* maze = new Maze(mazesize, 2 * mazesize);
		size_t leftlength = maze->getWallHugLength(true);
		size_t rightlength = maze->getWallHugLength(false);
		size_t largelength = std::max(leftlength, rightlength);
		maze->displayMaze(mazewin);
		printOutputs(textwin, convertedOutputs, numOutputs, moves, turnsWandering, mazesize, leftlength, rightlength, mazesSolved, mazesFailed, mazeaverage, moveTotals);
		turnsWandering = 0;
		moves = 0;
		for (size_t i = 0; i < 4; i++)
			moveTotals[i] = 0;
		while (true) {
			turnsWandering++;
			turn++;
			for (size_t i = 0; i < NUM_BOT_TURNS_PER_MOVE; i++) {
				maze->writeActorInputs(bot->h_inputs);
				bot->takeTurn();
			}
#ifdef OPPOSING_OUTPUTS
			convertedOutputs[0] = bot->h_outputs[0] - THOUGHT_BASELINE - 0.5;
			convertedOutputs[1] = bot->h_outputs[1] - THOUGHT_BASELINE - 0.5;
			convertedOutputs[2] = 0.5 + THOUGHT_BASELINE - bot->h_outputs[0];
			convertedOutputs[3] = 0.5 + THOUGHT_BASELINE - bot->h_outputs[1];
#endif
			if (maze->moveActor(mazewin, convertedOutputs, moveTotals)) {
				moves++;
			}
			if (maze->actorOnGoal()) {
				float extraPleasure = 0.0;
				numWinsInRow++;
				if (excessPain > 0.0f) {
					//extraPleasure = std::min(excessPain, numWinsInRow*1.0f);
					extraPleasure = excessPain;
				}
				excessPain -= extraPleasure + 1.0f;
#ifndef USE_EXTRA_PLEASURE
				extraPleasure = 0.0f;
#endif
				float pleasure = pleasurePainLengthMultiplier(turnsWandering)*(1.0f + extraPleasure);
				bot->givePleasurePain(pleasure);
				mazesSolved++;
				if (mazesize < 30)
					mazesize++;
				break;
			}
			else if (turnsWandering >= 10 * largelength) {
				numWinsInRow = 0;
				excessPain += 1.0f;
				float pain = -1.0f * pleasurePainLengthMultiplier(turnsWandering);
				bot->givePleasurePain(pain);
				mazesFailed++;
				if (mazesize > 2)
					mazesize--;
				break;
			}
		}
		bot->resetThoughts();
		bot->saveWeights("maze");
		mazeres[mazerespos] = mazesize;
		mazerespos = (mazerespos + 1) % NUM_MAZE_AVERAGE_RESULTS;
		mazeaverage = 0;
		if (mazesSolved + mazesFailed > NUM_MAZE_AVERAGE_RESULTS) {
			for (size_t i = 0; i < NUM_MAZE_AVERAGE_RESULTS; i++) {
				mazeaverage += mazeres[i];
			}
			mazeaverage /= NUM_MAZE_AVERAGE_RESULTS;
			avghistory << mazeaverage << std::endl;
		}
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

bool Maze::moveActor(WINDOW* win, float* outputs, size_t* moveTotals) {
	int dir = NONE;
	float maxout = -9999;
	for (size_t i = 0; i < 4; i++) {
		if (outputs[i] > maxout) {
			dir = (int)i;
			maxout = outputs[i];
		}
	}
	moveTotals[dir]++;
	if (directionOpen(actorPosition, dir)) {
		mvwaddch(win, (int)(2 * actorPosition.y + 1), (int)(2 * actorPosition.x + 1), ' ');
		actorPosition = getAdjacentPosition(actorPosition, dir);
		mvwaddch(win, (int)(2 * actorPosition.y + 1), (int)(2 * actorPosition.x + 1), '@');

		wrefresh(win);
		return true;
	}
	return false;
}

bool Maze::actorOnGoal() {
	return positionsSame(actorPosition, goalPosition);
}

bool positionsSame(Position p1, Position p2) {
	return p1.x == p2.x && p1.y == p2.y;
}

void printOutputs(WINDOW* win, float* outputs, size_t numOutputs, size_t moves, size_t turn, size_t level, size_t leftlength, size_t rightlength, size_t mazesSolved, size_t mazesFailed, float mazeaverage, size_t* moveTotals) {
	werase(win);
	std::stringstream ss;
	ss << moves << "/" << turn << "(";
	if (turn != 0)
		ss << 1.0f*moves / turn;
	ss << ")";
	mvwprintw(win, 0, 0, ss.str().c_str());

	size_t allMoveTotal = 0;
	for (size_t i = 0; i < 4; i++) {
		allMoveTotal += moveTotals[i];
	}

	for (size_t i = 0; i < 4; i++) {
		ss.clear();
		ss.str("");
		if (i == 0)
			ss << "Down: ";
		else if (i == 1)
			ss << "Right: ";
		else if (i == 2)
			ss << "Up: ";
		else if (i == 3)
			ss << "Left: ";
		ss << outputs[i];
		ss << " (" << moveTotals[i] << "; " << 1.0f*moveTotals[i] / allMoveTotal << ")";

		mvwprintw(win, i + 1, 0, ss.str().c_str());
	}

	std::stringstream lss;
	lss << "Maze Level: " << level;
	mvwprintw(win, 6, 0, lss.str().c_str());

	lss.clear();
	lss.str("");
	lss << "L: " << leftlength;
	mvwprintw(win, 7, 0, lss.str().c_str());

	lss.clear();
	lss.str("");
	lss << "R: " << rightlength;
	mvwprintw(win, 7, 7, lss.str().c_str());

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