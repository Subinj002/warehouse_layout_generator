"""
AI models for warehouse layout generation and optimization.

This module contains machine learning models and algorithms used to generate 
efficient warehouse layouts based on various inputs and constraints.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.cluster import KMeans
import joblib
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/models')


class LayoutGAN:
    """Generative Adversarial Network for warehouse layout generation."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 1), 
                 latent_dim: int = 100):
        """
        Initialize the Layout GAN model.
        
        Args:
            input_shape: Shape of the layout grid (height, width, channels)
            latent_dim: Dimension of the latent space
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
    def _build_generator(self) -> models.Model:
        """
        Build the generator model.
        
        Returns:
            A Keras model that generates warehouse layouts
        """
        noise_input = layers.Input(shape=(self.latent_dim,))
        
        # First dense layer
        x = layers.Dense(8 * 8 * 128, use_bias=False)(noise_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((8, 8, 128))(x)
        
        # Transposed convolutions to upscale
        x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Output layer
        output = layers.Conv2D(self.input_shape[2], (4, 4), padding='same', activation='tanh')(x)
        
        model = models.Model(noise_input, output)
        logger.info(f"Generator model built with input shape: {self.latent_dim}")
        return model
    
    def _build_discriminator(self) -> models.Model:
        """
        Build the discriminator model.
        
        Returns:
            A Keras model that discriminates between real and generated layouts
        """
        input_img = layers.Input(shape=self.input_shape)
        
        x = layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same')(input_img)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Flatten()(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(input_img, output)
        model.compile(loss='binary_crossentropy', 
                     optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     metrics=['accuracy'])
        
        logger.info(f"Discriminator model built with input shape: {self.input_shape}")
        return model
    
    def _build_gan(self) -> models.Model:
        """
        Build the combined GAN model.
        
        Returns:
            A Keras model that represents the full GAN
        """
        # Freeze the discriminator for generator training
        self.discriminator.trainable = False
        
        # Build the GAN by connecting generator and discriminator
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)
        
        model = models.Model(gan_input, gan_output)
        model.compile(loss='binary_crossentropy', 
                     optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
        
        logger.info("Combined GAN model built")
        return model
    
    def train(self, real_layouts: np.ndarray, epochs: int = 10000, 
              batch_size: int = 32, save_interval: int = 1000):
        """
        Train the GAN on real warehouse layouts.
        
        Args:
            real_layouts: Array of real warehouse layouts
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_interval: Interval for saving model checkpoints
        """
        # Rescale real layouts to [-1, 1]
        real_layouts = real_layouts.astype('float32')
        real_layouts = (real_layouts - 127.5) / 127.5
        
        # Labels for real and fake layouts
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, real_layouts.shape[0], batch_size)
            real_batch = real_layouts[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            # Log progress
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]} | G Loss: {g_loss}")
            
            # Save model
            if epoch % save_interval == 0:
                self.save_model(f"model_epoch_{epoch}")
    
    def generate_layout(self, num_layouts: int = 1) -> np.ndarray:
        """
        Generate warehouse layouts.
        
        Args:
            num_layouts: Number of layouts to generate
            
        Returns:
            Array of generated layouts
        """
        noise = np.random.normal(0, 1, (num_layouts, self.latent_dim))
        generated_layouts = self.generator.predict(noise)
        
        # Rescale from [-1, 1] to [0, 255]
        generated_layouts = (generated_layouts + 1) * 127.5
        generated_layouts = generated_layouts.astype(np.uint8)
        
        return generated_layouts
    
    def save_model(self, model_name: str = "layout_gan"):
        """
        Save the model to disk.
        
        Args:
            model_name: Name of the saved model
        """
        if not os.path.exists(DEFAULT_MODEL_PATH):
            os.makedirs(DEFAULT_MODEL_PATH)
            
        self.generator.save(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}_generator.h5"))
        self.discriminator.save(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}_discriminator.h5"))
        logger.info(f"Model saved as {model_name} in {DEFAULT_MODEL_PATH}")
    
    def load_model(self, model_name: str = "layout_gan"):
        """
        Load the model from disk.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.generator = models.load_model(
                os.path.join(DEFAULT_MODEL_PATH, f"{model_name}_generator.h5"))
            self.discriminator = models.load_model(
                os.path.join(DEFAULT_MODEL_PATH, f"{model_name}_discriminator.h5"))
            
            # Rebuild GAN
            self.gan = self._build_gan()
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise


class WarehouseLayoutClassifier:
    """Classification model for evaluating warehouse layouts."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 1)):
        """
        Initialize the classifier model.
        
        Args:
            input_shape: Shape of the input layout images
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self) -> models.Model:
        """
        Build the classifier model.
        
        Returns:
            A Keras model for layout classification
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')  # 3 classes: Efficient, Average, Inefficient
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        logger.info("Layout classifier model built")
        return model
    
    def train(self, layouts: np.ndarray, labels: np.ndarray, 
              validation_split: float = 0.2, epochs: int = 20, batch_size: int = 32):
        """
        Train the classifier on labeled warehouse layouts.
        
        Args:
            layouts: Array of warehouse layouts
            labels: Array of layout labels (one-hot encoded)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Normalize layouts to [0, 1]
        layouts = layouts.astype('float32') / 255.0
        
        history = self.model.fit(
            layouts, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        logger.info(f"Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        return history
    
    def evaluate_layout(self, layout: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a warehouse layout.
        
        Args:
            layout: Warehouse layout to evaluate
            
        Returns:
            Dictionary with class probabilities
        """
        if len(layout.shape) == 3:  # Single layout
            layout = np.expand_dims(layout, axis=0)
            
        # Normalize layout to [0, 1]
        layout = layout.astype('float32') / 255.0
        
        predictions = self.model.predict(layout)
        
        # Convert predictions to dictionary
        classes = ['Efficient', 'Average', 'Inefficient']
        result = {class_name: float(pred) for class_name, pred in zip(classes, predictions[0])}
        
        return result
    
    def save_model(self, model_name: str = "layout_classifier"):
        """
        Save the model to disk.
        
        Args:
            model_name: Name of the saved model
        """
        if not os.path.exists(DEFAULT_MODEL_PATH):
            os.makedirs(DEFAULT_MODEL_PATH)
            
        self.model.save(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.h5"))
        logger.info(f"Model saved as {model_name} in {DEFAULT_MODEL_PATH}")
    
    def load_model(self, model_name: str = "layout_classifier"):
        """
        Load the model from disk.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.model = models.load_model(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.h5"))
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise


class LayoutClusteringModel:
    """Clustering model for grouping similar warehouse layouts."""
    
    def __init__(self, n_clusters: int = 5):
        """
        Initialize the clustering model.
        
        Args:
            n_clusters: Number of clusters to form
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.trained = False
        
    def preprocess_layouts(self, layouts: np.ndarray) -> np.ndarray:
        """
        Preprocess layouts for clustering.
        
        Args:
            layouts: Array of warehouse layouts
            
        Returns:
            Flattened and normalized layouts
        """
        # Flatten layouts
        flattened = layouts.reshape(layouts.shape[0], -1)
        
        # Normalize to [0, 1]
        normalized = flattened.astype('float32') / 255.0
        
        return normalized
    
    def train(self, layouts: np.ndarray):
        """
        Train the clustering model.
        
        Args:
            layouts: Array of warehouse layouts
        """
        preprocessed = self.preprocess_layouts(layouts)
        self.model.fit(preprocessed)
        self.trained = True
        
        logger.info(f"Clustering model trained with {self.n_clusters} clusters")
        
    def predict_cluster(self, layout: np.ndarray) -> int:
        """
        Predict the cluster for a layout.
        
        Args:
            layout: Warehouse layout to classify
            
        Returns:
            Cluster ID
        """
        if not self.trained:
            logger.error("Model not trained yet")
            raise ValueError("Model must be trained before making predictions")
            
        # Ensure layout is properly shaped
        if len(layout.shape) == 3:  # Single layout
            layout = np.expand_dims(layout, axis=0)
            
        preprocessed = self.preprocess_layouts(layout)
        cluster_id = self.model.predict(preprocessed)[0]
        
        return int(cluster_id)
    
    def get_cluster_centroids(self) -> np.ndarray:
        """
        Get the centroid of each cluster.
        
        Returns:
            Array of cluster centroids
        """
        if not self.trained:
            logger.error("Model not trained yet")
            raise ValueError("Model must be trained before accessing centroids")
            
        return self.model.cluster_centers_
    
    def save_model(self, model_name: str = "layout_clustering"):
        """
        Save the model to disk.
        
        Args:
            model_name: Name of the saved model
        """
        if not os.path.exists(DEFAULT_MODEL_PATH):
            os.makedirs(DEFAULT_MODEL_PATH)
            
        joblib.dump(self.model, os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.pkl"))
        logger.info(f"Model saved as {model_name} in {DEFAULT_MODEL_PATH}")
    
    def load_model(self, model_name: str = "layout_clustering"):
        """
        Load the model from disk.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.model = joblib.load(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.pkl"))
            self.trained = True
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise


class ReinforcementLayoutOptimizer:
    """Reinforcement learning model for optimizing warehouse layouts."""
    
    def __init__(self, 
                 state_shape: Tuple[int, int, int] = (64, 64, 1),
                 action_space: int = 9):
        """
        Initialize the RL model.
        
        Args:
            state_shape: Shape of the state space (layout grid)
            action_space: Size of the action space
        """
        self.state_shape = state_shape
        self.action_space = action_space
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> models.Model:
        """
        Build the deep Q-network model.
        
        Returns:
            A Keras model for reinforcement learning
        """
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.state_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_space, activation='linear')
        ])
        
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss='mse')
        
        logger.info("Reinforcement learning model built")
        return model
    
    def update_target_model(self):
        """Update the target model with the weights of the main model."""
        self.target_model.set_weights(self.model.get_weights())
        logger.info("Target model updated")
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_space)
        
        # Ensure state is properly shaped
        if len(state.shape) == 3:  # Single state
            state = np.expand_dims(state, axis=0)
            
        # Normalize state to [0, 1]
        state = state.astype('float32') / 255.0
        
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def train(self, 
              states: np.ndarray, 
              actions: np.ndarray, 
              rewards: np.ndarray, 
              next_states: np.ndarray, 
              dones: np.ndarray,
              gamma: float = 0.95,
              batch_size: int = 32):
        """
        Train the model on a batch of experiences.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor
            batch_size: Training batch size
        """
        # Normalize states to [0, 1]
        states = states.astype('float32') / 255.0
        next_states = next_states.astype('float32') / 255.0
        
        # Get current Q values
        current_q = self.model.predict(states)
        
        # Get target Q values
        target_q = current_q.copy()
        
        # Get next Q values from target model
        next_q = self.target_model.predict(next_states)
        
        # Update target Q values
        for i in range(batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + gamma * np.max(next_q[i])
        
        # Train the model
        history = self.model.fit(states, target_q, verbose=0)
        
        return history.history['loss'][0]
    
    def save_model(self, model_name: str = "layout_rl"):
        """
        Save the model to disk.
        
        Args:
            model_name: Name of the saved model
        """
        if not os.path.exists(DEFAULT_MODEL_PATH):
            os.makedirs(DEFAULT_MODEL_PATH)
            
        self.model.save(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.h5"))
        logger.info(f"Model saved as {model_name} in {DEFAULT_MODEL_PATH}")
    
    def load_model(self, model_name: str = "layout_rl"):
        """
        Load the model from disk.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.model = models.load_model(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.h5"))
            self.target_model = models.load_model(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.h5"))
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise


class HeatmapGenerator:
    """Model for generating heatmaps of warehouse activity or efficiency."""
    
    def __init__(self, grid_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the heatmap generator.
        
        Args:
            grid_size: Size of the warehouse grid
        """
        self.grid_size = grid_size
        self.model = self._build_model()
        
    def _build_model(self) -> models.Model:
        """
        Build the heatmap generation model.
        
        Returns:
            A Keras model for generating heatmaps
        """
        # Input: warehouse layout and traffic data
        layout_input = layers.Input(shape=(*self.grid_size, 1), name='layout_input')
        traffic_input = layers.Input(shape=(*self.grid_size, 1), name='traffic_input')
        
        # Concatenate inputs
        combined = layers.Concatenate()([layout_input, traffic_input])
        
        # Process combined input
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(combined)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Upsampling
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        
        # Output heatmap
        heatmap_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='heatmap')(x)
        
        model = models.Model(inputs=[layout_input, traffic_input], outputs=heatmap_output)
        model.compile(optimizer='adam', loss='mse')
        
        logger.info("Heatmap generator model built")
        return model
    
    def train(self, 
              layouts: np.ndarray, 
              traffic_data: np.ndarray, 
              target_heatmaps: np.ndarray,
              validation_split: float = 0.2, 
              epochs: int = 20, 
              batch_size: int = 8):
        """
        Train the heatmap generator model.
        
        Args:
            layouts: Array of warehouse layouts
            traffic_data: Array of traffic data
            target_heatmaps: Array of target heatmaps
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Normalize inputs to [0, 1]
        layouts = layouts.astype('float32') / 255.0
        traffic_data = traffic_data.astype('float32') / 255.0
        target_heatmaps = target_heatmaps.astype('float32') / 255.0
        
        history = self.model.fit(
            [layouts, traffic_data], 
            target_heatmaps,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}")
        return history
    
    def generate_heatmap(self, layout: np.ndarray, traffic_data: np.ndarray) -> np.ndarray:
        """
        Generate a heatmap for a warehouse layout and traffic data.
        
        Args:
            layout: Warehouse layout
            traffic_data: Traffic data for the warehouse
            
        Returns:
            Generated heatmap
        """
        # Ensure inputs are properly shaped
        if len(layout.shape) == 2:
            layout = np.expand_dims(layout, axis=(0, -1))
        elif len(layout.shape) == 3 and layout.shape[-1] != 1:
            layout = np.expand_dims(layout, axis=0)
        elif len(layout.shape) == 3 and layout.shape[-1] == 1:
            layout = np.expand_dims(layout, axis=0)
            
        if len(traffic_data.shape) == 2:
            traffic_data = np.expand_dims(traffic_data, axis=(0, -1))
        elif len(traffic_data.shape) == 3 and traffic_data.shape[-1] != 1:
            traffic_data = np.expand_dims(traffic_data, axis=0)
        elif len(traffic_data.shape) == 3 and traffic_data.shape[-1] == 1:
            traffic_data = np.expand_dims(traffic_data, axis=0)
            
        # Normalize inputs to [0, 1]
        layout = layout.astype('float32') / 255.0
        traffic_data = traffic_data.astype('float32') / 255.0
        
        heatmap = self.model.predict([layout, traffic_data])
        
        # Convert back to [0, 255] range
        heatmap = (heatmap * 255).astype(np.uint8)
        
        return heatmap[0, :, :, 0]  # Return as 2D array
    
    def save_model(self, model_name: str = "heatmap_generator"):
        """
        Save the model to disk.
        
        Args:
            model_name: Name of the saved model
        """
        if not os.path.exists(DEFAULT_MODEL_PATH):
            os.makedirs(DEFAULT_MODEL_PATH)
            
        self.model.save(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.h5"))
        logger.info(f"Model saved as {model_name} in {DEFAULT_MODEL_PATH}")
    
    def load_model(self, model_name: str = "heatmap_generator"):
        """
        Load the model from disk.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.model = models.load_model(os.path.join(DEFAULT_MODEL_PATH, f"{model_name}.h5"))
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise