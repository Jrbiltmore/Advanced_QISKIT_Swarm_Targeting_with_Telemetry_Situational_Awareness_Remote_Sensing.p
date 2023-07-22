# haptic_feedback.py
# This file contains functions for generating haptic feedback signals to convey information to drone operators using tactile cues.

def generate_haptic_feedback(target_distance, threat_level, terrain_impact):
    """
    Generate haptic feedback signals based on telemetry, threat, and terrain data.

    Parameters:
        target_distance (float): The distance to the target.
        threat_level (int): The threat level (0 for no threat, 1 for potential threat).
        terrain_impact (float): The impact of terrain on targeting.

    Returns:
        str: A haptic feedback signal indicating the feedback to the drone operator.
    """
    # Convert the target distance to a vibration intensity value
    vibration_intensity = map_distance_to_vibration_intensity(target_distance)

    # Determine the haptic feedback based on threat level and terrain impact
    if threat_level == 1:
        feedback = "Strong Vibration - Potential Threat!"
    else:
        feedback = "Vibration - Targeting on Track."

    # Incorporate the impact of terrain on the haptic feedback
    if terrain_impact >= 0.5:
        feedback += " Caution - High Terrain Impact!"

    # Add the vibration intensity to the feedback
    feedback += f" Vibration Intensity: {vibration_intensity}%"

    return feedback

def map_distance_to_vibration_intensity(distance):
    """
    Map the target distance to a vibration intensity value.

    Parameters:
        distance (float): The distance to the target.

    Returns:
        int: The vibration intensity value (0-100) based on the distance.
    """
    # Some mapping function to convert distance to vibration intensity
    # For example, if distance is in the range [0, 100], map it to the range [0, 100]
    vibration_intensity = int((distance / 100) * 100)
    return vibration_intensity
