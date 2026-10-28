classdef card < handle
    % CARD
    %
    %
    %
    % John W. Miller
    % 14-Mar-2017
    
    properties
        name
        value
    end
    
    methods(Static)
        
        % Replace the default plot() function
        function plot()
            fprintf('\nHELLO TEST TEST\n\n')
        end
        
        function disp()
            fprintf('Card name: %s\n',obj.name)
        end
        
    end % End static methods
        
    methods
        % Constructor
        function obj = card(Name,Value)
            obj.name  = Name;
            obj.value = Value;
        end
        
        %% Plotting Functions
        
        % Plot the step cycle
        function plot2(obj)
            disp('TEST')            
        end
                
    end
end