from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from stable_baselines3.common.base_class import BaseAlgorithm

console = Console()


def evaluate_model(
    model: BaseAlgorithm, env: gym.Env, num_episodes: int = 5, deterministic: bool = True, verbose: bool = True
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Evaluate a trained model on an environment.

    Args:
        model (BaseAlgorithm): The trained model to evaluate
        env (gym.Env): The environment to evaluate on
        num_episodes (int, optional): Number of episodes to run. Defaults to 10.
        deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.
        verbose (bool, optional): Whether to print detailed information. Defaults to True.

    Returns:
        Tuple[List[float], List[Dict[str, Any]]]: A tuple containing:
            - episode_rewards: List of total rewards for each episode
            - episode_results: List of detailed results for each episode
    """
    if verbose:
        console.print(
            f"[bold blue]Evaluating {model.__class__.__name__} model for {num_episodes} episodes...[/bold blue]"
        )

    all_episode_results = []
    total_rewards = []

    for episode in track(range(num_episodes), description="Running episodes..."):
        if verbose:
            console.print(f"[yellow]--- Episode {episode + 1}/{num_episodes} ---[/yellow]")

        # Initialize episode tracking
        episode_results = {
            "episode": episode + 1,
            "steps": 0,
            "final_value": 0,
            "total_reward": 0,
            "initial_value": 0,
            "actions_taken": {},
            "detailed_actions": [],
            "rewards_per_step": [],
        }

        try:
            # Reset environment
            obs, info = env.reset()
            episode_results["initial_value"] = info.get("portfolio_value", 100000.0)

            terminated = False
            truncated = False
            episode_reward = 0.0
            step_count = 0

            while not terminated and not truncated:
                # Get action from model
                action, _states = model.predict(obs, deterministic=deterministic)

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_results["rewards_per_step"].append(reward)

                # Track action information
                action_decoded = info.get("action_decoded", {})
                action_type = action_decoded.get("type", "Unknown")
                amount_pct = action_decoded.get("amount_pct", 0.0)
                price_modifier = action_decoded.get("price_modifier", 1.0)
                is_valid = not action_decoded.get("invalid_action_attempt", False)

                current_price = info.get("current_price", 0.0)
                portfolio_value = info.get("portfolio_value", 0.0)

                # Store detailed action information
                detailed_action = {
                    "step": step_count,
                    "action_type": action_type,
                    "amount_pct": amount_pct,
                    "price_modifier": price_modifier,
                    "is_valid": is_valid,
                    "current_price": current_price,
                    "portfolio_value": portfolio_value,
                    "reward": reward,
                }
                episode_results["detailed_actions"].append(detailed_action)

                # Update action counts
                if isinstance(action_type, str):
                    episode_results["actions_taken"][action_type] = (
                        episode_results["actions_taken"].get(action_type, 0) + 1
                    )

                # Progress reporting
                if verbose and step_count % 50 == 0:
                    validity_status = "[green]✓[/green]" if is_valid else "[red]✗[/red]"
                    console.print(
                        f"  Step {step_count}: {validity_status} {action_type}, "
                        f"Price: [cyan]${current_price:.2f}[/cyan], "
                        f"Portfolio: [green]${portfolio_value:.2f}[/green], "
                        f"Reward: [yellow]{reward:.4f}[/yellow]"
                    )

                step_count += 1

            # Store final episode results
            episode_results["steps"] = step_count
            episode_results["final_value"] = info.get("portfolio_value", 0.0)
            episode_results["total_reward"] = episode_reward

            # Calculate returns
            initial_val = episode_results["initial_value"]
            final_val = episode_results["final_value"]
            return_pct = ((final_val - initial_val) / initial_val) * 100 if initial_val > 0 else 0

            if verbose:
                console.print(f"[bold green]Episode {episode + 1} Results:[/bold green]")
                console.print(f"  Total Steps: [cyan]{step_count}[/cyan]")
                console.print(f"  Initial Portfolio Value: [green]${initial_val:.2f}[/green]")
                console.print(f"  Final Portfolio Value: [green]${final_val:.2f}[/green]")

                return_color = "green" if return_pct >= 0 else "red"
                console.print(f"  Total Return: [{return_color}]{return_pct:.2f}%[/{return_color}]")
                console.print(f"  Total Reward: [yellow]{episode_reward:.2f}[/yellow]")

                # Print action distribution in a table
                if episode_results["actions_taken"]:
                    action_table = Table(title="Action Distribution", show_header=True)
                    action_table.add_column("Action", style="cyan")
                    action_table.add_column("Count", justify="right", style="green")
                    action_table.add_column("Percentage", justify="right", style="yellow")

                    for action_name, count in episode_results["actions_taken"].items():
                        percentage = (count / step_count) * 100 if step_count > 0 else 0
                        action_table.add_row(action_name, str(count), f"{percentage:.1f}%")

                    console.print(action_table)

            total_rewards.append(episode_reward)
            all_episode_results.append(episode_results)

        except Exception as e:
            console.print(f"[red]Error during episode {episode + 1}: {e}[/red]")
            total_rewards.append(0.0)
            episode_results["error"] = str(e)
            all_episode_results.append(episode_results)

    # Print overall summary
    if verbose:
        console.print("\n[bold magenta]=== Overall Evaluation Summary ===[/bold magenta]")

        summary_table = Table(title="Summary Statistics", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Episodes", str(num_episodes))
        summary_table.add_row("Average Reward", f"{np.mean(total_rewards):.2f}")
        summary_table.add_row("Reward Std Dev", f"{np.std(total_rewards):.2f}")
        summary_table.add_row("Best Episode Reward", f"{max(total_rewards):.2f}")
        summary_table.add_row("Worst Episode Reward", f"{min(total_rewards):.2f}")

        # Calculate average return
        valid_episodes = [ep for ep in all_episode_results if "error" not in ep]
        if valid_episodes:
            avg_return = np.mean(
                [((ep["final_value"] - ep["initial_value"]) / ep["initial_value"]) * 100 for ep in valid_episodes]
            )
            return_color = "green" if avg_return >= 0 else "red"
            summary_table.add_row("Average Return", f"[{return_color}]{avg_return:.2f}%[/{return_color}]")

        console.print(summary_table)

    return total_rewards, all_episode_results


def evaluate_multiple_models(
    models: Dict[str, BaseAlgorithm],
    env: gym.Env,
    num_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[str, Tuple[List[float], List[Dict[str, Any]]]]:
    """
    Evaluate multiple models on the same environment.

    Args:
        models (Dict[str, BaseAlgorithm]): A dictionary of model names and their corresponding instances.
        env (gym.Env): The environment to evaluate the models on.
        num_episodes (int, optional): The number of episodes to run for each model. Defaults to 10.
        deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.
        verbose (bool, optional): Whether to print detailed information. Defaults to True.

    Returns:
        Dict[str, Tuple[List[float], List[Dict[str, Any]]]]: A dictionary mapping model names
        to their evaluation results.
    """
    results = {}

    for model_name, model in models.items():
        if verbose:
            console.print(f"\n[bold blue]{'='*60}[/bold blue]")
            console.print(f"[bold blue]EVALUATING {model_name}[/bold blue]")
            console.print(f"[bold blue]{'='*60}[/bold blue]")

        rewards, episodes = evaluate_model(
            model=model, env=env, num_episodes=num_episodes, deterministic=deterministic, verbose=verbose
        )

        results[model_name] = (rewards, episodes)

    return results


def compare_model_performance(
    evaluation_results: Dict[str, Tuple[List[float], List[Dict[str, Any]]]], verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance metrics across multiple models.

    Args:
        evaluation_results (Dict[str, Tuple[List[float], List[Dict[str, Any]]]]):
        The evaluation results for each model.
        verbose (bool, optional): Whether to print the comparison table. Defaults to True.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping model names to their performance metrics.
    """
    performance_metrics = {}

    for model_name, (rewards, episodes) in evaluation_results.items():
        valid_episodes = [ep for ep in episodes if "error" not in ep]

        if valid_episodes:
            # Calculate metrics
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            avg_return = np.mean(
                [((ep["final_value"] - ep["initial_value"]) / ep["initial_value"]) * 100 for ep in valid_episodes]
            )
            std_return = np.std(
                [((ep["final_value"] - ep["initial_value"]) / ep["initial_value"]) * 100 for ep in valid_episodes]
            )
            avg_steps = np.mean([ep["steps"] for ep in valid_episodes])

            performance_metrics[model_name] = {
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "avg_return_pct": avg_return,
                "std_return_pct": std_return,
                "avg_steps": avg_steps,
                "num_episodes": len(valid_episodes),
            }

    if verbose and performance_metrics:
        console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
        console.print("[bold magenta]MODEL PERFORMANCE COMPARISON[/bold magenta]")
        console.print(f"[bold magenta]{'='*80}[/bold magenta]")

        # Create comparison table
        comparison_table = Table(title="Model Performance Comparison", show_header=True, header_style="bold magenta")
        comparison_table.add_column("Model", style="cyan", no_wrap=True)
        comparison_table.add_column("Avg Reward", justify="right", style="green")
        comparison_table.add_column("Std Reward", justify="right", style="yellow")
        comparison_table.add_column("Avg Return %", justify="right", style="bold green")
        comparison_table.add_column("Std Return %", justify="right", style="yellow")
        comparison_table.add_column("Avg Steps", justify="right", style="blue")
        comparison_table.add_column("Episodes", justify="right", style="cyan")

        for model_name, metrics in performance_metrics.items():
            return_color = "green" if metrics["avg_return_pct"] >= 0 else "red"
            comparison_table.add_row(
                model_name,
                f"{metrics['avg_reward']:.2f}",
                f"{metrics['std_reward']:.2f}",
                f"[{return_color}]{metrics['avg_return_pct']:.2f}%[/{return_color}]",
                f"{metrics['std_return_pct']:.2f}%",
                f"{metrics['avg_steps']:.0f}",
                str(metrics["num_episodes"]),
            )

        console.print(comparison_table)

    return performance_metrics


def get_action_statistics(episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract action statistics from episode results.

    Args:
        episode_results (List[Dict[str, Any]]): A list of episode result dictionaries.

    Returns:
        Dict[str, Any]: A dictionary containing action statistics.
    """
    # Aggregate all actions
    all_actions = {}
    total_steps = 0

    for episode in episode_results:
        if "error" not in episode:
            total_steps += episode["steps"]
            for action_type, count in episode.get("actions_taken", {}).items():
                all_actions[action_type] = all_actions.get(action_type, 0) + count

    # Calculate statistics
    action_stats = {"total_steps": total_steps, "action_counts": all_actions, "action_percentages": {}}

    if total_steps > 0:
        for action_type, count in all_actions.items():
            action_stats["action_percentages"][action_type] = (count / total_steps) * 100

    return action_stats
