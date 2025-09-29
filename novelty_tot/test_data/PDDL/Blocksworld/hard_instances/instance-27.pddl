

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b a)
(ontable c)
(on d e)
(on e c)
(ontable f)
(clear b)
(clear f)
)
(:goal
(and
(on a d)
(on d e)
(on e b)
(on f c))
)
)


