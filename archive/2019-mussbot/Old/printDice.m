function printDice(allDice,varargin)
% PRINTDICE displays a set of dice in the command prompt
%
%	INPUT
%       dice: [n_rolls x n_dice]
%
% John W. Miller
% 05-Mar-2017

if nargin > 1
    mask_busts = varargin{1};
else
    mask_busts = [];
end

if nargin > 2
    mask_fills = varargin{2};
else
    mask_fills = [];
end

fprintf('\n')
for n_roll = 1:size(allDice,1);
    dice = allDice(n_roll,:);
    n_dice = sum(~isnan(dice));
    
    if n_dice > 0
        % Convert the dice values to strings
        diceStrings = cell(n_dice,1);
        for n = 1:n_dice
            diceStrings{n} = getDiceString(dice(n));
        end
        
        % Assemble the string to print
        stringToPrint = [];
        rows = [1:5; 6:10; 11:15];
        for r = 1:3
            for n = 1:n_dice
                stringToPrint = [stringToPrint diceStrings{n}(rows(r,:))];
            end
            stringToPrint = [stringToPrint '\n'];
        end
        
        % Print the dice!
        if ~isempty(mask_busts)
            if mask_busts(n_roll)
                cprintf('red',[stringToPrint, '\n'])
            else
                fprintf([stringToPrint, '\n'])
            end
        else
            fprintf([stringToPrint, '\n'])
        end
    else
        fprintf('Dice are all NaN\n')
    end
end

end % End of main

%% INTERNAL FUNCTION
function diceString = getDiceString(num)
switch num
    case 1
        diceString = sprintf('|   || * ||   |');
    case 2
        diceString = sprintf('|*  ||   ||  *|');
    case 3
        diceString = sprintf('|*  || * ||  *|');
    case 4
        diceString = sprintf('|* *||   ||* *|');
    case 5
        diceString = sprintf('|* *|| * ||* *|');
    case 6
        diceString = sprintf('|* *||* *||* *|');
    otherwise
        error('Gotta roll a D6!')
end
end


