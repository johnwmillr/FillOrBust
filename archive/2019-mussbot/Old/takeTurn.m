function total_score = takeTurn(n_dice_to_stop_at)
% TAKETURN
%
%	INPUT
%
%
%
%	OUTPUT
%
%
% John W. Miller
% 10-Jan-2017

% Take the first roll
n_rolls = 10;
dice = rollDice(n_rolls);
scores = [];

% Roll until there are n_dice_remaining
while any(sum(~isnan(dice),2) > n_dice_to_stop_at) & ~all(scores==0) | isempty(scores)
    
    % Determine score for the roll and how many dice to roll with next time
    [score, n_dice_to_roll_next] = calcScore(dice);
    scores = [scores score];
    
    % Roll one set of dice, removing pointers as you go
    dice = rollDice(n_rolls,n_dice_to_roll_next);
end

% Tally up the final score
if any(scores==0)
    total_score = 0; % Bust!
else
    total_score = sum(scores);
end

end % End of main