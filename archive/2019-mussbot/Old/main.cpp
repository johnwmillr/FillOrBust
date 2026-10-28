// http://www.learncpp.com/cpp-tutorial/6-x-chapter-6-comprehensive-quiz/

#include <iostream>
#include <array>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include "CardGame.h"


enum die
{
  DIE_ONE,
  DIE_TWO,
  DIE_THREE,
  DIE_FOUR,
  DIE_FIVE,
  DIE_SIX
};


//void rollDice(std::array<die dice




int main()
{
  // Random seeding
  srand(static_cast<unsigned int>(time(0))); // set initial seed value to system clock
  rand(); // Throw out the first rand value produced

  //std::array<die,6> dice;
  

  return 0;
}
