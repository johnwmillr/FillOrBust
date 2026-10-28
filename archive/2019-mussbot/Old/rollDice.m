function rolls = rollDice(n_turns,n_dice_to_roll,n_dice_to_stop_at)
% ROLLDICE
%
%	INPUT
%
%
%
%	OUTPUT
%
%
% John W. Miller
% 09-Jan-2017

n_sides = 6;
n_dice = 6;
if nargin < 2 || isempty(n_dice_to_roll)
    n_dice_to_roll = 6*ones(n_turns,1);    
end
if nargin < 3
    n_dice_to_stop_at = 0;
end

% Roll dice
rolls = nan(n_turns,n_dice);
if all(n_dice_to_roll==6)    
    % Roll six dice for all rolls
    rolls = randi(n_sides,[n_turns n_dice]);    
else
    % If user supplied numbers of dice to roll    
    for n_roll = 1:n_turns
        if n_dice_to_roll(n_roll) ~= 0 && n_dice_to_roll(n_roll) > n_dice_to_stop_at
            rolls(n_roll,1:n_dice_to_roll(n_roll)) = randi(n_sides, [1 n_dice_to_roll(n_roll)]);
        end
    end
end

% Sort the dice
rolls = sort(rolls,2);

end % End of main