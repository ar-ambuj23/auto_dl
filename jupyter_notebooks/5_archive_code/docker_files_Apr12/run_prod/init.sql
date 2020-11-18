CREATE DATABASE configdb;
use configdb;

CREATE TABLE `config_table` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `model_name` varchar(255) DEFAULT NULL,
  `model_type` varchar(255) DEFAULT NULL,
  `list_of_features` text,
  `target_column` varchar(255) DEFAULT NULL,
  `RMSE` float DEFAULT NULL,
  `Percent_Error` float DEFAULT NULL,
  `accuracy` float DEFAULT NULL,
  `prec` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `model_name` (`model_name`)
);
