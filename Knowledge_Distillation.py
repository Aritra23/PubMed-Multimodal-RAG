import torch
import torch.nn.functional as F

def get_soft_targets(teacher_model, inputs, temperature=2.0):
    outputs = teacher_model(inputs)
    soft_targets = F.softmax(outputs / temperature, dim=1)
    return soft_targets

def distillation_loss_function(student_outputs, teacher_outputs, alpha=0.5, temperature=2.0):
    # Cross-Entropy Loss
    ce_loss = F.cross_entropy(student_outputs, torch.argmax(teacher_outputs, dim=1))
    
    # Kullback-Leibler Divergence
    distillation_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction='batchmean'
    )
    
    # Combined Loss
    loss = alpha * ce_loss + (1 - alpha) * distillation_loss * (temperature ** 2)
    return loss

def train_student_model(teacher_model, student_model, data_loader, learning_rate=1e-4, epochs=10, temperature=2.0, alpha=0.5):
    teacher_model.eval()
    student_model.train()
    
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in data_loader:
            inputs, _ = batch
            
            # Get teacher outputs
            with torch.no_grad():
                teacher_outputs = get_soft_targets(teacher_model, inputs, temperature)
            
            # Get student outputs
            student_outputs = student_model(inputs)
            
            # Compute loss
            loss = distillation_loss_function(student_outputs, teacher_outputs, alpha, temperature)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(data_loader)}')
    return student_model