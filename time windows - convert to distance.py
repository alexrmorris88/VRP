from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

actual_distance = []

pick_distance = [0, 0, 0]
drop_distance = [682, 870, 1211]

actual_distance = [682, 870, 1211]

time_windows = [
    (10, 17),  # 2
    (8, 17),  # 3
    (13, 17),  # 4
]


# =============================================================================
# Changing the time windows to AM and PM
# =============================================================================
def time_windows_to_am_pm(time):
    """
    convert time to am and pm
    """
    if time[0] <= 12 and time[1] <= 12:
        time = f'{time[0]} am to {time[1]} am'
    elif time[0] <= 12 and time[1] > 12:
        time = f'{time[0]} am to {time[1] - 12} pm'
    elif time[0] > 12 and time[1] > 12:
        time = f'{time[0] - 12} pm to {time[1] - 12} pm'
    return time


# =============================================================================
# Changing the time windows to a 24h period
# =============================================================================
def time_windows_to_distance():
    """ Try and figureout how to change the time windows based on distance? **Do time zones affect this? """
    time_window = []
    for distance, time in zip(drop_distance, time_windows):
        if distance < 1_000:
            time = ((time[0] - 8) * 100, (time[1] - 8) * 100)
            time_window.append(time)
        elif distance >= 1_000:
            time = (((time[0] - 8) + 24) * 100, ((time[1] - 8) + 24) * 100)
            time_window.append(time)
    return time_window


time_windows_to_distance_var = time_windows_to_distance()

""" then figureout the average speed to the time it would take to get there (for the time windows) """

""" add in the actual distance with the time windows, maybe a second manager? """


# =============================================================================
# converting driving waiting time to distance
# =============================================================================
def drop_dist():
    """ formula adds extra distance for loads outside of 24 hours. """
    for drop in drop_distance:
        if drop >= 1_000:
            drop_distance.remove(drop)
            drop2 = drop + 1_000
            drop_distance.append(drop2)

    for number, bounds in zip(drop_distance, time_windows_to_distance_var):
        if (bounds[0] <= number <= bounds[1]) == False:
            if (bounds[0] - number) < 800:
                new_number = number + (bounds[0] - number)
                drop_distance.remove(number)
                drop_distance.append(new_number)
            else:
                number

    return drop_distance


drop = drop_dist()


def distance_matrix():
    create_distance_matrix_pick = pick_distance
    create_distance_matrix_drop = actual_distance

    create_distance_matrix_pick_depot = create_distance_matrix_pick[:1]
    create_distance_matrix = create_distance_matrix_pick_depot + create_distance_matrix_drop

    create_distance_matrix = np.array(create_distance_matrix)
    distance_matrix = np.abs(create_distance_matrix - create_distance_matrix.reshape(-1, 1))
    drop_len = create_distance_matrix
    dummy = [0 for i in range(len(drop_len))]
    distance_matrix = np.c_[dummy, distance_matrix]
    dummy_2 = [0 for i in range(len(drop_len) + 1)]
    dummy_2 = np.array(dummy_2)
    distance_matrix = np.vstack((dummy_2, distance_matrix))

    return distance_matrix


def time_matrix():
    create_distance_matrix_pick = pick_distance
    create_distance_matrix_drop = drop

    create_distance_matrix_pick_depot = create_distance_matrix_pick[:1]
    create_distance_matrix = create_distance_matrix_pick_depot + create_distance_matrix_drop

    create_distance_matrix = np.array(create_distance_matrix)
    distance_matrix = np.abs(create_distance_matrix - create_distance_matrix.reshape(-1, 1))
    drop_len = create_distance_matrix
    dummy = [0 for i in range(len(drop_len))]
    distance_matrix = np.c_[dummy, distance_matrix]
    dummy_2 = [0 for i in range(len(drop_len) + 1)]
    dummy_2 = np.array(dummy_2)
    distance_matrix = np.vstack((dummy_2, distance_matrix))

    return distance_matrix


def create_data_model():
    distance_matrix_ = distance_matrix()
    time_matrix_ = time_matrix()
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix_
    data['time_matrix'] = time_matrix_
    data['actual_distance'] = [(0), (0)]
    data['actual_distance'] += actual_distance
    data['actual_time_windows'] = [(0, 0), (0, 0)]
    data['actual_time_windows'] += time_windows
    data['time_windows'] = [(0, 0), (0, 0)]
    data['time_windows'] += time_windows_to_distance_var
    data['weight'] = [0, 0, 25_000, 10_000, 10_000]
    data['customer'] = [
        "dummy",
        "FBD",
        "Anjou, QC",
        "Ottawa, ON",
        "Toronto, ON"]
    data['num_vehicles'] = 2
    data['vehicle_capacities'] = [42_000 for i in range(data['num_vehicles'])]
    data['start'] = [1 for i in range(data['num_vehicles'])]
    data['end'] = [0 for i in range(data['num_vehicles'])]
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            plan_output += '{} ({}) ({:,})-> '.format(data['customer'][node_index],
                                                      time_windows_to_am_pm(data['actual_time_windows'][node_index]),
                                                      data['weight'][node_index])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            distance = data['actual_distance'][node_index]
        plan_output += '{}\n'.format('Route End')
        plan_output += 'Distance of the route: {}m  Actual distance: {}\n'.format(route_distance, distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}, Weight: {:,} lbs'.format(solution.ObjectiveValue(), sum(data['weight'])))


def main():
    distance_matrix_ = distance_matrix()
    max_dist = np.sum(distance_matrix_)
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['start'], data['end'])
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30,  # allow waiting time
        4000,  # maximum time per vehicle (Think about it as max distance per vehicle)
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except start and end.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx != 0 and 1:
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Add time window constraints for each vehicle start node.
    start_idx = 1
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][start_idx][0],
            data['time_windows'][start_idx][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

        # Create and register a transit callback.

    def distance_callback(from_index, to_index):
        """Return the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Return the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['weight'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Number of locations per vehicle
    def num_of_locations(from_index):
        """Return 1 for any locations except depot."""
        # Convert from routing variable Index to user NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return 1 if (from_node != 0) else 0;

    counter_callback_index = routing.RegisterUnaryTransitCallback(num_of_locations)

    routing.AddDimensionWithVehicleCapacity(
        counter_callback_index,
        0,  # null slack
        [(30) for i in range(data['num_vehicles'])],  # maximum locations per vehicle
        True,  # start cumul to zero
        'num_of_locations')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
    search_parameters.solution_limit = 100_000_000
    search_parameters.time_limit.FromSeconds(1)
    search_parameters.log_search = False

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print('No solution found !')


if __name__ == '__main__':
    main()